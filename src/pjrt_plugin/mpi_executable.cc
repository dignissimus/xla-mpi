#include "pjrt_plugin/mpi_executable.h"
#include "pjrt_plugin/mpi_collectives.h"
#include "pjrt_plugin/mpi_collectives_async.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "tsl/platform/logging.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/xla_data.pb.h"

namespace xla_mpi {

namespace {

namespace se = ::stream_executor;

constexpr char kDeviceMemoryKind[] = "device";

PJRT_Buffer_Type MlirElementTypeToPjrtType(mlir::Type type) {
    return pjrt::ConvertToPjRtBufferType(xla::ConvertMlirTypeToPrimitiveType(type));
}

struct ProgramInfo {
    int64_t num_replicas = 1;
    int64_t num_partitions = 1;
    std::vector<int64_t> device_assignment_flat;  // [replica * num_partitions + partition]

    int64_t my_rank = 0;
    int64_t my_replica = 0;
    int64_t my_partition = 0;
};

std::optional<xla::ReductionKind> MatchReductionBody(mlir::Region& body) {
    if (!body.hasOneBlock()) return std::nullopt;
    mlir::Block& block = body.front();
    if (block.getNumArguments() != 2) return std::nullopt;

    auto return_op = mlir::dyn_cast_or_null<mlir::stablehlo::ReturnOp>(block.getTerminator());
    if (!return_op || return_op.getNumOperands() != 1) return std::nullopt;

    mlir::Operation* reduce_op = return_op.getOperand(0).getDefiningOp();
    if (!reduce_op || reduce_op->getBlock() != &block) return std::nullopt;

    llvm::SmallPtrSet<mlir::Value, 2> args = {block.getArgument(0), block.getArgument(1)};
    auto uses_both_args = [&](mlir::Value lhs, mlir::Value rhs) {
        return args.contains(lhs) && args.contains(rhs);
    };

    if (auto op = mlir::dyn_cast<mlir::stablehlo::AddOp>(reduce_op)) {
        if (uses_both_args(op.getLhs(), op.getRhs())) return xla::ReductionKind::SUM;
    } else if (auto op = mlir::dyn_cast<mlir::stablehlo::MulOp>(reduce_op)) {
        if (uses_both_args(op.getLhs(), op.getRhs())) return xla::ReductionKind::PRODUCT;
    } else if (auto op = mlir::dyn_cast<mlir::stablehlo::MinOp>(reduce_op)) {
        if (uses_both_args(op.getLhs(), op.getRhs())) return xla::ReductionKind::MIN;
    } else if (auto op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(reduce_op)) {
        if (uses_both_args(op.getLhs(), op.getRhs())) return xla::ReductionKind::MAX;
    }
    return std::nullopt;
}

std::optional<xla::ReductionKind> MatchReductionBodyOrWarn(mlir::Region& body,
                                                            const char* match_fn_name) {
    std::optional<xla::ReductionKind> kind = MatchReductionBody(body);
    if (!kind) {
        LOG(WARNING) << match_fn_name
                     << ": reduction body is not a plain SUM/PRODUCT/MIN/MAX over both block "
                        "arguments, which xampi does not support for async operations, so will "
                        "use synchronous communication.";
    }
    return kind;
}

mlir::NamedAttribute MakeRequestBufferAttr(mlir::OpBuilder& builder) {
    auto* buffer = new MpiRequestBuffer();
    return builder.getNamedAttr("request_buffer",
                                builder.getI64IntegerAttr(reinterpret_cast<int64_t>(buffer)));
}

mlir::NamedAttribute MakeReduceScatterRequestBufferAttr(mlir::OpBuilder& builder) {
    auto* buffer = new MpiReduceScatterRequestBuffer();
    return builder.getNamedAttr("request_buffer",
                                builder.getI64IntegerAttr(reinterpret_cast<int64_t>(buffer)));
}

ProcessGroupStrategy ResolveReplicaFamilyStrategy(int64_t channel_id, bool use_global_device_ids) {
    if (channel_id <= 0) return ProcessGroupStrategy::kCrossReplica;
    return use_global_device_ids ? ProcessGroupStrategy::kFlattenedIds
                                 : ProcessGroupStrategy::kCrossReplicaAndPartition;
}

// channel_id <= 0 selects kCrossReplica, otherwise kCrossPartition -- for
// ops that only carry channel_handle, not use_global_device_ids
// (CollectivePermute/AllToAll/Send/Recv) -- see spec.md's "cross_partition"
// section.
ProcessGroupStrategy ResolvePermuteFamilyStrategy(int64_t channel_id) {
    return channel_id <= 0 ? ProcessGroupStrategy::kCrossReplica : ProcessGroupStrategy::kCrossPartition;
}

int64_t GetChannelId(std::optional<mlir::stablehlo::ChannelHandleAttr> channel_handle) {
    return channel_handle.has_value() ? channel_handle->getHandle() : 0;
}

bool AppendProcessGroupAttrs(mlir::OpBuilder& builder, mlir::Attribute groups_attr,
                             ProcessGroupStrategy strategy, const ProgramInfo& program_info,
                             bool include_group_metadata, bool include_my_rank,
                             llvm::SmallVectorImpl<mlir::NamedAttribute>& attrs) {
    auto groups = mlir::dyn_cast<mlir::DenseIntElementsAttr>(groups_attr);
    if (!groups) return false;

    auto device_assignment_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(program_info.device_assignment_flat.size())}, builder.getI64Type());
    mlir::Attribute device_assignment_attr =
        mlir::DenseElementsAttr::get(device_assignment_type, llvm::ArrayRef<int64_t>(
                                                                  program_info.device_assignment_flat));

    attrs.push_back(builder.getNamedAttr("groups", groups));
    if (include_group_metadata) {
        int64_t num_groups = groups.getType().getShape()[0];
        attrs.push_back(builder.getNamedAttr("num_groups", builder.getI64IntegerAttr(num_groups)));
    }
    attrs.push_back(builder.getNamedAttr(
        "process_group_strategy", builder.getI32IntegerAttr(static_cast<int32_t>(strategy))));
    attrs.push_back(builder.getNamedAttr("num_partitions",
                                        builder.getI64IntegerAttr(program_info.num_partitions)));
    attrs.push_back(builder.getNamedAttr("device_assignment", device_assignment_attr));
    if (include_my_rank) {
        attrs.push_back(
            builder.getNamedAttr("my_rank", builder.getI64IntegerAttr(program_info.my_rank)));
    }
    attrs.push_back(
        builder.getNamedAttr("my_replica", builder.getI64IntegerAttr(program_info.my_replica)));
    attrs.push_back(
        builder.getNamedAttr("my_partition", builder.getI64IntegerAttr(program_info.my_partition)));
    return true;
}

bool IsPlainDenseGroups(mlir::Attribute attr, const char* match_fn_name) {
    if (mlir::isa<mlir::DenseIntElementsAttr>(attr)) return true;
    LOG(WARNING) << match_fn_name
                 << ": replica_groups/source_target_pairs is not a plain "
                    "DenseIntElementsAttr, likely Shardy mesh-axes form, which "
                    "xampi does not support for async operations, so will use "
                    "synchronous communication.";
    return false;
}

std::optional<xla::ReductionKind> MatchAllReduceForAsyncRewrite(mlir::stablehlo::AllReduceOp op) {
    if (!IsPlainDenseGroups(op.getReplicaGroups(), "MatchAllReduceForAsyncRewrite")) return std::nullopt;
    return MatchReductionBodyOrWarn(op.getComputation(), "MatchAllReduceForAsyncRewrite");
}

// TODO: Currently splits the multi-operand form into independent segments.
// Consider if it's ever beneficial to group them and use MPI_WaitAll
void RewriteAsAsyncAllReduce(mlir::stablehlo::AllReduceOp op, xla::ReductionKind reduction_kind,
                             const ProgramInfo& program_info, int32_t& next_group_tag) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolveReplicaFamilyStrategy(channel_id, op.getUseGlobalDeviceIds());

    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
        mlir::Value operand = op.getOperands()[i];
        mlir::Type result_type = op.getResult(i).getType();
        mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

        llvm::SmallVector<mlir::NamedAttribute, 9> config_attrs = {
            builder.getNamedAttr("reduction_kind",
                                builder.getI32IntegerAttr(static_cast<int32_t>(reduction_kind))),
            builder.getNamedAttr("group_tag", builder.getI32IntegerAttr(next_group_tag++)),
            request_buffer,
        };
        if (!AppendProcessGroupAttrs(builder, op.getReplicaGroups(), strategy, program_info,
                                    /*include_group_metadata=*/true, /*include_my_rank=*/false,
                                    config_attrs)) {
            LOG(FATAL) << "MatchAllReduceForAsyncRewrite verified replica_groups is "
                          "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                          "such.";
        }
        auto backend_config = builder.getDictionaryAttr(config_attrs);

        llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.allreduce_start")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("api_version", api_version),
            builder.getNamedAttr("backend_config", backend_config),
        };
        auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
            loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

        auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
            context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
        llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.allreduce_done")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("api_version", api_version),
            builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
            builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
        };
        auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
            loc, mlir::TypeRange{result_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

        op.getResult(i).replaceAllUsesWith(done_op.getResult(0));
    }
    op.erase();
}

std::optional<xla::ReductionKind> MatchReduceScatterForAsyncRewrite(mlir::stablehlo::ReduceScatterOp op) {
    if (!IsPlainDenseGroups(op.getReplicaGroups(), "MatchReduceScatterForAsyncRewrite")) return std::nullopt;
    return MatchReductionBodyOrWarn(op.getComputation(), "MatchReduceScatterForAsyncRewrite");
}

void RewriteAsAsyncReduceScatter(mlir::stablehlo::ReduceScatterOp op, xla::ReductionKind reduction_kind,
                                 const ProgramInfo& program_info, int32_t group_tag) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    mlir::Value operand = op.getOperand();
    mlir::Type result_type = op.getResult().getType();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolveReplicaFamilyStrategy(channel_id, op.getUseGlobalDeviceIds());
    mlir::NamedAttribute request_buffer = MakeReduceScatterRequestBufferAttr(builder);

    llvm::SmallVector<mlir::NamedAttribute, 9> config_attrs = {
        builder.getNamedAttr("reduction_kind",
                            builder.getI32IntegerAttr(static_cast<int32_t>(reduction_kind))),
        builder.getNamedAttr("scatter_dimension", builder.getI64IntegerAttr(op.getScatterDimension())),
        builder.getNamedAttr("group_tag", builder.getI32IntegerAttr(group_tag)),
        request_buffer,
    };
    if (!AppendProcessGroupAttrs(builder, op.getReplicaGroups(), strategy, program_info,
                                /*include_group_metadata=*/true, /*include_my_rank=*/true,
                                config_attrs)) {
        LOG(FATAL) << "MatchReduceScatterForAsyncRewrite verified replica_groups is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.reduce_scatter_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", backend_config),
    };
    auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

    auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
        context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
        builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.reduce_scatter_done")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
        builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
    };
    auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

    op.getResult().replaceAllUsesWith(done_op.getResult(0));
    op.erase();
}

bool MatchAllGatherForAsyncRewrite(mlir::stablehlo::AllGatherOp op) {
    return IsPlainDenseGroups(op.getReplicaGroups(), "MatchAllGatherForAsyncRewrite");
}

// Same shape as RewriteAsAsyncAllReduce, targeting xla_mpi.allgather_start/
// done instead; no reduction kind involved, but threads all_gather_dim
// through for the same reason ReduceScatter threads scatter_dimension.
//
// TODO: Currently splits the multi-operand form into independent segments.
// Consider if it's ever beneficial to group them and use MPI_WaitAll
void RewriteAsAsyncAllGather(mlir::stablehlo::AllGatherOp op, const ProgramInfo& program_info,
                             int32_t& next_group_tag) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolveReplicaFamilyStrategy(channel_id, op.getUseGlobalDeviceIds());

    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
        mlir::Value operand = op.getOperands()[i];
        mlir::Type result_type = op.getResult(i).getType();
        mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

        llvm::SmallVector<mlir::NamedAttribute, 9> config_attrs = {
            builder.getNamedAttr("all_gather_dim", builder.getI64IntegerAttr(op.getAllGatherDim())),
            builder.getNamedAttr("group_tag", builder.getI32IntegerAttr(next_group_tag++)),
            request_buffer,
        };
        if (!AppendProcessGroupAttrs(builder, op.getReplicaGroups(), strategy, program_info,
                                    /*include_group_metadata=*/true, /*include_my_rank=*/false,
                                    config_attrs)) {
            LOG(FATAL) << "MatchAllGatherForAsyncRewrite verified replica_groups is "
                          "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                          "such.";
        }
        auto backend_config = builder.getDictionaryAttr(config_attrs);

        llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.allgather_start")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("api_version", api_version),
            builder.getNamedAttr("backend_config", backend_config),
        };
        auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
            loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

        auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
            context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
        llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.allgather_done")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("api_version", api_version),
            builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
            builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
        };
        auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
            loc, mlir::TypeRange{result_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

        op.getResult(i).replaceAllUsesWith(done_op.getResult(0));
    }
    op.erase();
}

bool MatchCollectivePermuteForAsyncRewrite(mlir::stablehlo::CollectivePermuteOp) { return true; }

void RewriteAsAsyncCollectivePermute(mlir::stablehlo::CollectivePermuteOp op,
                                     const ProgramInfo& program_info) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    mlir::Value operand = op.getOperand();
    mlir::Type result_type = op.getResult().getType();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolvePermuteFamilyStrategy(channel_id);
    mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

    llvm::SmallVector<mlir::NamedAttribute, 8> config_attrs = {request_buffer};
    if (!AppendProcessGroupAttrs(builder, op.getSourceTargetPairs(), strategy, program_info,
                                /*include_group_metadata=*/false, /*include_my_rank=*/true,
                                config_attrs)) {
        LOG(FATAL) << "MatchCollectivePermuteForAsyncRewrite verified source_target_pairs is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.collective_permute_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", backend_config),
    };
    auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

    // recv_buffer is Done's sole operand -- Done's sole result aliases it.
    auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
        context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.collective_permute_done")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
        builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
    };
    auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

    op.getResult().replaceAllUsesWith(done_op.getResult(0));
    op.erase();
}

void RewriteCollectivesAsAsync(mlir::func::FuncOp entry, const ProgramInfo& program_info) {
    int32_t next_group_tag = 0;

    llvm::SmallVector<std::pair<mlir::stablehlo::AllReduceOp, xla::ReductionKind>> all_reduce_rewrites;
    entry.walk([&](mlir::stablehlo::AllReduceOp op) {
        if (std::optional<xla::ReductionKind> kind = MatchAllReduceForAsyncRewrite(op)) {
            all_reduce_rewrites.push_back({op, *kind});
        }
    });
    for (auto& [op, kind] : all_reduce_rewrites) {
        RewriteAsAsyncAllReduce(op, kind, program_info, next_group_tag);
    }

    llvm::SmallVector<std::pair<mlir::stablehlo::ReduceScatterOp, xla::ReductionKind>>
        reduce_scatter_rewrites;
    entry.walk([&](mlir::stablehlo::ReduceScatterOp op) {
        if (std::optional<xla::ReductionKind> kind = MatchReduceScatterForAsyncRewrite(op)) {
            reduce_scatter_rewrites.push_back({op, *kind});
        }
    });
    for (auto& [op, kind] : reduce_scatter_rewrites) {
        RewriteAsAsyncReduceScatter(op, kind, program_info, next_group_tag++);
    }

    llvm::SmallVector<mlir::stablehlo::AllGatherOp> all_gather_rewrites;
    entry.walk([&](mlir::stablehlo::AllGatherOp op) {
        if (MatchAllGatherForAsyncRewrite(op)) all_gather_rewrites.push_back(op);
    });
    for (mlir::stablehlo::AllGatherOp op : all_gather_rewrites) {
        RewriteAsAsyncAllGather(op, program_info, next_group_tag);
    }

    llvm::SmallVector<mlir::stablehlo::CollectivePermuteOp> collective_permute_rewrites;
    entry.walk([&](mlir::stablehlo::CollectivePermuteOp op) {
        if (MatchCollectivePermuteForAsyncRewrite(op)) collective_permute_rewrites.push_back(op);
    });
    for (mlir::stablehlo::CollectivePermuteOp op : collective_permute_rewrites) {
        RewriteAsAsyncCollectivePermute(op, program_info);
    }
}

mlir::func::FuncOp FindEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;
    for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
        if (func_op.getName() == "main") return func_op;
        if (!entry) entry = func_op;
    }
    return entry;
}

}  // namespace

std::shared_ptr<MpiExecutable> MpiExecutable::Create(const std::string& format, const char* code,
                                                     size_t code_size, const char* compile_options,
                                                     size_t compile_options_size) {
    std::shared_ptr<MpiExecutable> exe(new MpiExecutable());

    if (format != "mlir" && format != "hlo" && format != "hlo_with_config") {
        exe->error_ = "Unknown program format: " + format;
        return exe;
    }

    mlir::MLIRContext context;
    context.disableMultithreading();
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_or =
        xla::ParseMlirModuleString(absl::string_view(code, code_size), context);
    if (!module_or.ok()) {
        exe->error_ = "Failed to parse StableHLO program: " + std::string(module_or.status().message());
        return exe;
    }
    mlir::ModuleOp module = **module_or;

    mlir::func::FuncOp entry = FindEntryFunction(module);
    if (!entry) {
        exe->error_ = "No entry function found in module";
        return exe;
    }
    if (!entry.getBody().hasOneBlock()) {
        exe->error_ = "Only single-block entry functions are supported for now";
        return exe;
    }
    mlir::Block& block = entry.getBody().front();

    xla::ExecutableBuildOptions build_options;
    if (compile_options != nullptr && compile_options_size > 0) {
        xla::CompileOptionsProto proto;
        if (!proto.ParseFromArray(compile_options, static_cast<int>(compile_options_size))) {
            exe->error_ = "Failed to parse CompileOptionsProto";
            return exe;
        }
        absl::StatusOr<xla::CompileOptions> parsed_options = xla::CompileOptions::FromProto(proto);
        if (!parsed_options.ok()) {
            exe->error_ =
                "Failed to convert CompileOptionsProto: " + std::string(parsed_options.status().message());
            return exe;
        }
        build_options = parsed_options->executable_build_options;
    }

    exe->num_replicas_ = build_options.num_replicas();
    exe->num_partitions_ = build_options.num_partitions();
    if (!build_options.has_device_assignment()) {
        exe->error_ = "MpiExecutable::Create: compile options must include a device assignment";
        return exe;
    }
    exe->device_assignment_ = build_options.device_assignment();

    ProgramInfo program_info;
    program_info.num_replicas = exe->num_replicas_;
    program_info.num_partitions = exe->num_partitions_;
    program_info.device_assignment_flat.reserve(
        static_cast<size_t>(exe->num_replicas_ * exe->num_partitions_));
    for (int64_t r = 0; r < exe->num_replicas_; ++r) {
        for (int64_t p = 0; p < exe->num_partitions_; ++p) {
            program_info.device_assignment_flat.push_back(exe->device_assignment_(r, p));
        }
    }

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    program_info.my_rank = my_rank;
    program_info.my_replica = -1;
    program_info.my_partition = -1;
    for (int64_t r = 0; r < exe->num_replicas_ && program_info.my_replica < 0; ++r) {
        for (int64_t p = 0; p < exe->num_partitions_; ++p) {
            int64_t idx = r * exe->num_partitions_ + p;
            if (program_info.device_assignment_flat[idx] == my_rank) {
                program_info.my_replica = r;
                program_info.my_partition = p;
                break;
            }
        }
    }

    RewriteCollectivesAsAsync(entry, program_info);

    mlir::func::ReturnOp return_op;
    for (mlir::Operation& op : block) {
        if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) return_op = ret;
    }
    if (!return_op) {
        exe->error_ = "Entry function has no return op";
        return exe;
    }

    std::vector<xla::Shape> input_shapes;
    for (mlir::Value arg : block.getArguments()) {
        auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
        if (!tensor_type) {
            exe->error_ = "Only ranked tensor inputs are supported";
            return exe;
        }
        PJRT_Buffer_Type dtype = MlirElementTypeToPjrtType(tensor_type.getElementType());
        xla::PrimitiveType prim_type = pjrt::ConvertFromPjRtBufferType(dtype);
        if (dtype == PJRT_Buffer_Type_INVALID || prim_type == xla::PRIMITIVE_TYPE_INVALID) {
            exe->error_ = "Unsupported input element type";
            return exe;
        }
        std::vector<int64_t> shape_vec = tensor_type.getShape().vec();
        exe->input_info_.push_back(ArgInfo{static_cast<int>(dtype), shape_vec});
        input_shapes.push_back(xla::ShapeUtil::MakeShape(prim_type, shape_vec));
    }

    for (mlir::Value result : return_op.getOperands()) {
        auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
        if (!tensor_type) {
            exe->error_ = "Only ranked tensor outputs are supported";
            return exe;
        }
        PJRT_Buffer_Type dtype = MlirElementTypeToPjrtType(tensor_type.getElementType());
        if (dtype == PJRT_Buffer_Type_INVALID) {
            exe->error_ = "Unsupported output element type";
            return exe;
        }
        exe->output_info_.push_back(OutputInfo{static_cast<int>(dtype), tensor_type.getShape().vec()});
    }

    xla::XlaComputation computation;
    absl::Status convert_status = xla::MlirToXlaComputation(
        module, computation, /*use_tuple_args=*/false, /*return_tuple=*/true,
        /*exec_build_options=*/nullptr);
    if (!convert_status.ok()) {
        exe->error_ = "Failed to convert StableHLO to HLO: " + std::string(convert_status.message());
        return exe;
    }

    absl::StatusOr<xla::LocalClient*> client_or = xla::ClientLibrary::GetOrCreateLocalClient();
    if (!client_or.ok()) {
        exe->error_ = "Failed to get XLA CPU client: " + std::string(client_or.status().message());
        return exe;
    }

    std::vector<const xla::Shape*> argument_layouts;
    argument_layouts.reserve(input_shapes.size());
    for (const xla::Shape& shape : input_shapes) argument_layouts.push_back(&shape);

    absl::StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>> executables =
        (*client_or)->Compile(computation, argument_layouts, build_options);
    if (!executables.ok()) {
        exe->error_ =
            "Failed to compile StableHLO on XLA's CPU backend: " + std::string(executables.status().message());
        return exe;
    }
    if (executables->empty()) {
        exe->error_ = "XLA compilation produced no executable";
        return exe;
    }

    exe->executable_ = std::move((*executables)[0]);
    exe->client_ = *client_or;

    for (const OutputInfo& info : exe->output_info_) {
        exe->output_types_.push_back(static_cast<PJRT_Buffer_Type>(info.dtype));
        exe->output_dim_sizes_.push_back(info.shape.size());
        exe->output_dims_.insert(exe->output_dims_.end(), info.shape.begin(), info.shape.end());
    }
    exe->output_memory_kinds_.assign(exe->output_info_.size(), kDeviceMemoryKind);
    exe->output_memory_kind_sizes_.assign(exe->output_info_.size(), std::strlen(kDeviceMemoryKind));
    exe->fingerprint_ = std::to_string(reinterpret_cast<uintptr_t>(exe->executable_.get()));

    exe->valid_ = true;
    return exe;
}

MpiExecutable::~MpiExecutable() = default;

bool MpiExecutable::IsValid() const { return valid_; }
std::string MpiExecutable::error() const { return error_; }
size_t MpiExecutable::num_outputs() const { return output_info_.size(); }

MpiExecuteResult MpiExecutable::Execute(const std::vector<MpiBuffer*>& inputs) {
    MpiExecuteResult result;
    if (!valid_) {
        result.error_message = error_.empty() ? "Executable is not valid" : error_;
        return result;
    }

    if (inputs.size() != input_info_.size()) {
        result.error_message = "Expected " + std::to_string(input_info_.size()) +
                               " input(s), got " + std::to_string(inputs.size());
        return result;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i] || static_cast<int>(inputs[i]->dtype()) != input_info_[i].dtype ||
            inputs[i]->shape() != input_info_[i].shape) {
            result.error_message = "Input " + std::to_string(i) + " has an unexpected dtype or shape";
            return result;
        }
    }

    std::vector<xla::ExecutionInput> arguments;
    arguments.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        MpiBuffer* buf = inputs[i];
        xla::PrimitiveType prim_type =
            pjrt::ConvertFromPjRtBufferType(static_cast<PJRT_Buffer_Type>(input_info_[i].dtype));
        xla::Shape shape = xla::ShapeUtil::MakeShape(prim_type, input_info_[i].shape);
        xla::ExecutionInput input(shape);
        input.SetUnownedBuffer(
            xla::ShapeIndex(),
            xla::MaybeOwningDeviceAddress(se::DeviceAddressBase(buf->data(), buf->byte_size())));
        arguments.push_back(std::move(input));
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    xla::cpu::CpuExecutableRunOptions cpu_run_options;
    cpu_run_options.set_collectives(&xla_mpi::GetMpiSingleton().collectives());

    xla::ExecutableRunOptions run_options;
    run_options.set_allocator(client_->backend().memory_allocator());
    run_options.set_cpu_executable_run_options(&cpu_run_options);
    run_options.set_device_ordinal(rank);
    run_options.set_physical_device_ordinal(0);
    run_options.set_device_assignment(&device_assignment_);
    absl::StatusOr<xla::ExecutionOutput> output = executable_->RunAsync(std::move(arguments), run_options);
    if (!output.ok()) {
        result.error_message = "XLA execution failed: " + std::string(output.status().message());
        return result;
    }

    xla::ShapedBuffer released = output->ConsumeResult().release();
    std::vector<std::unique_ptr<MpiBuffer>> owned_outputs;
    owned_outputs.reserve(output_info_.size());
    for (size_t i = 0; i < output_info_.size(); ++i) {
        const OutputInfo& info = output_info_[i];
        se::DeviceAddressBase src = released.buffer(xla::ShapeIndex({static_cast<int64_t>(i)}));
        auto out_buf = MpiBuffer::AdoptFromXla(src.opaque(), static_cast<PJRT_Buffer_Type>(info.dtype), info.shape);
        if (src.size() != out_buf->byte_size()) {
            result.error_message = "Output " + std::to_string(i) + " size mismatch after execution";
            return result;
        }
        owned_outputs.push_back(std::move(out_buf));
    }

    for (auto& out : owned_outputs) result.buffers.push_back(out.release());
    return result;
}

}  // namespace xla_mpi

// ============================================================================
// C API
// ============================================================================

PJRT_Error* MPI_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPI_Executable_Name(PJRT_Executable_Name_Args* args) {
    static const char* name = "mpi_executable";
    args->executable_name = name;
    args->executable_name_size = 14;
    return nullptr;
}

PJRT_Error* MPI_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args) {
    args->num_replicas = static_cast<int>(args->executable->executable->num_replicas());
    return nullptr;
}

PJRT_Error* MPI_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args) {
    args->num_partitions = static_cast<int>(args->executable->executable->num_partitions());
    return nullptr;
}

PJRT_Error* MPI_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
    args->num_outputs = args->executable->executable->num_outputs();
    return nullptr;
}

PJRT_Error* MPI_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    args->size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPI_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    return MakeError("MPI_Executable_GetCostAnalysis not yet implemented");
}

PJRT_Error* MPI_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    const xla_mpi::MpiExecutable& exe = *args->executable->executable;
    args->num_outputs = exe.output_memory_kinds().size();
    args->memory_kinds = exe.output_memory_kinds().data();
    args->memory_kind_sizes = exe.output_memory_kind_sizes().data();
    return nullptr;
}

PJRT_Error* MPI_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args) {
    return MakeError("MPI_Executable_OptimizedProgram not yet implemented");
}

PJRT_Error* MPI_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
    return MakeError("MPI_Executable_Serialize not yet implemented");
}

PJRT_Error* MPI_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    if (args->executable) {
        delete args->executable->executable;
    }
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    PJRT_Executable* source = args->loaded_executable->executable;
    PJRT_Executable* independent = new PJRT_Executable();
    independent->executable = source->executable;
    independent->client = source->client;
    independent->owned_by_loaded = false;
    args->executable = independent;
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
    args->addressable_devices = args->executable->addressable_devices.data();
    args->num_addressable_devices = args->executable->addressable_devices.size();
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args) {
    args->executable->deleted = true;
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
    args->is_deleted = args->executable->deleted;
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    PJRT_LoadedExecutable* loaded = args->executable;
    if (!loaded || !loaded->executable || !loaded->executable->executable) {
        return MakeError("MPI_LoadedExecutable_Execute: invalid executable");
    }
    xla_mpi::MpiExecutable& mpi_exe = *loaded->executable->executable;

    if (args->num_devices != 1) {
        return MakeError(
            "MPI_LoadedExecutable_Execute: this phase only supports single-device "
            "execution per rank (num_devices must be 1)");
    }

    std::vector<xla_mpi::MpiBuffer*> inputs;
    inputs.reserve(args->num_args);
    for (size_t i = 0; i < args->num_args; ++i) {
        PJRT_Buffer* buf = args->argument_lists[0][i];
        if (!buf || !buf->buffer) {
            return MakeError("MPI_LoadedExecutable_Execute: null or deleted input buffer");
        }
        inputs.push_back(buf->buffer.get());
    }

    xla_mpi::MpiExecuteResult result = mpi_exe.Execute(inputs);
    if (!result.error_message.empty()) {
        return MakeError("MPI_LoadedExecutable_Execute: " + result.error_message);
    }

    for (size_t i = 0; i < result.buffers.size(); ++i) {
        PJRT_Buffer* out = new PJRT_Buffer();
        out->buffer.reset(result.buffers[i]);
        out->client = loaded->client;
        if (!loaded->addressable_devices.empty()) {
            out->device = loaded->addressable_devices[0];
            out->memory = out->device->default_memory;
        }
        args->output_lists[0][i] = out;
    }

    if (args->device_complete_events != nullptr) {
        args->device_complete_events[0] = new PJRT_Event{.ready = true};
    }

    return nullptr;
}

PJRT_Error* MPI_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
    return MakeError("MPI_Executable_DeserializeAndLoad not yet implemented");
}

PJRT_Error* MPI_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
    const std::string& fp = args->executable->executable->executable->fingerprint();
    args->executable_fingerprint = fp.c_str();
    args->executable_fingerprint_size = fp.size();
    return nullptr;
}

PJRT_Error* MPI_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
    const xla_mpi::MpiExecutable& exe = *args->executable->executable;
    args->output_types = const_cast<PJRT_Buffer_Type*>(exe.output_types().data());
    args->num_output_types = exe.output_types().size();
    return nullptr;
}

PJRT_Error* MPI_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    const xla_mpi::MpiExecutable& exe = *args->executable->executable;
    args->num_outputs = exe.output_dim_sizes().size();
    args->dims = exe.output_dims().data();
    args->dim_sizes = exe.output_dim_sizes().data();
    return nullptr;
}

PJRT_Error* MPI_Executable_GetCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    return MakeError("MPI_Executable_GetCompiledMemoryStats not yet implemented");
}

PJRT_Error* MPI_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    const std::string& fp = args->executable->executable->fingerprint();
    args->executable_fingerprint = fp.c_str();
    args->executable_fingerprint_size = fp.size();
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
    const xla::DeviceAssignment& device_assignment =
        args->executable->executable->executable->device_assignment();

    xla::DeviceAssignmentProto proto;
    device_assignment.Serialize(&proto);

    auto* serialized = new PJRT_DeviceAssignmentSerialized();
    if (!proto.SerializeToString(&serialized->bytes)) {
        delete serialized;
        return MakeError("Failed to serialize device assignment");
    }
    args->serialized_device_assignment = serialized;
    args->serialized_bytes = serialized->bytes.data();
    args->serialized_bytes_size = serialized->bytes.size();
    args->serialized_device_assignment_deleter =
        [](PJRT_DeviceAssignmentSerialized* da) { delete da; };
    return nullptr;
}
