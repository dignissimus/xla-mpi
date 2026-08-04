#include "pjrt_plugin/mlir_rewrite.h"
#include "pjrt_plugin/mpi_collectives_async.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "tsl/platform/logging.h"
#include "xla/core/collectives/reduction_kind.h"

namespace xla_mpi {

namespace {


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
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
                                /*include_group_metadata=*/true, /*include_my_rank=*/true, config_attrs)) {
        LOG(FATAL) << "MatchReduceScatterForAsyncRewrite verified replica_groups is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.reduce_scatter_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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
                                /*include_group_metadata=*/false, /*include_my_rank=*/true, config_attrs)) {
        LOG(FATAL) << "MatchCollectivePermuteForAsyncRewrite verified source_target_pairs is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.collective_permute_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", backend_config),
    };
    auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

    auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
        context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
        builder.getNamedAttr("call_target_name",
                            builder.getStringAttr("xla_mpi.collective_permute_done")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
        builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
    };
    auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

    op.getResult().replaceAllUsesWith(done_op.getResult(0));
    op.erase();
}

bool MatchAllToAllForAsyncRewrite(mlir::stablehlo::AllToAllOp op) {
    return IsPlainDenseGroups(op.getReplicaGroups(), "MatchAllToAllForAsyncRewrite");
}

// TODO: Currently splits the multi-operand form into independent segments.
// Consider if it's ever beneficial to group them and use MPI_WaitAll
void RewriteAsAsyncAllToAll(mlir::stablehlo::AllToAllOp op, const ProgramInfo& program_info) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolvePermuteFamilyStrategy(channel_id);

    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
        mlir::Value operand = op.getOperands()[i];
        mlir::Type result_type = op.getResult(i).getType();
        mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

        llvm::SmallVector<mlir::NamedAttribute, 9> config_attrs = {
            builder.getNamedAttr("split_dimension", builder.getI64IntegerAttr(op.getSplitDimension())),
            builder.getNamedAttr("concat_dimension", builder.getI64IntegerAttr(op.getConcatDimension())),
            builder.getNamedAttr("split_count", builder.getI64IntegerAttr(op.getSplitCount())),
            request_buffer,
        };
        if (!AppendProcessGroupAttrs(builder, op.getReplicaGroups(), strategy, program_info,
                                    /*include_group_metadata=*/true, /*include_my_rank=*/false,
                                    config_attrs)) {
            LOG(FATAL) << "MatchAllToAllForAsyncRewrite verified replica_groups is "
                          "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                          "such.";
        }
        auto backend_config = builder.getDictionaryAttr(config_attrs);

        llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.all_to_all_start")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
            builder.getNamedAttr("api_version", api_version),
            builder.getNamedAttr("backend_config", backend_config),
        };
        auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
            loc, mlir::TypeRange{result_type}, mlir::ValueRange{operand}, start_attrs);

        auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
            context, /*outputTupleIndices=*/{}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
        llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
            builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.all_to_all_done")),
            builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
            builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
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

bool MatchSendForAsyncRewrite(mlir::stablehlo::SendOp op) {
    if (op.getIsHostTransfer()) return false;
    if (op.getInputs().size() != 1) {
        LOG(WARNING) << "MatchSendForAsyncRewrite: multi-operand stablehlo.send shares one token "
                        "across all operands, which xampi does not support splitting for async "
                        "operations, so will use synchronous communication.";
        return false;
    }
    auto attr = op.getSourceTargetPairsAttr();
    if (!attr) {
        LOG(WARNING) << "MatchSendForAsyncRewrite: source_target_pairs is not set, so will use "
                        "synchronous communication.";
        return false;
    }
    return IsPlainDenseGroups(attr, "MatchSendForAsyncRewrite");
}

void RewriteAsAsyncSend(mlir::stablehlo::SendOp op, const ProgramInfo& program_info) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    mlir::Value operand = op.getInputs()[0];
    mlir::Value incoming_token = op.getToken();
    mlir::Type token_type = op.getToken().getType();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);
    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolvePermuteFamilyStrategy(channel_id);
    mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

    llvm::SmallVector<mlir::NamedAttribute, 8> config_attrs = {
        builder.getNamedAttr("channel_id", builder.getI64IntegerAttr(channel_id)),
        request_buffer,
    };
    if (!AppendProcessGroupAttrs(builder, op.getSourceTargetPairsAttr(), strategy, program_info,
                                /*include_group_metadata=*/false, /*include_my_rank=*/true, config_attrs)) {
        LOG(FATAL) << "MatchSendForAsyncRewrite verified source_target_pairs is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.send_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", backend_config),
    };
    auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{token_type}, mlir::ValueRange{operand, incoming_token}, start_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> done_attrs = {
        builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.send_done")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
    };
    auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{token_type}, mlir::ValueRange{start_op.getResult(0)}, done_attrs);

    op.getResult().replaceAllUsesWith(done_op.getResult(0));
    op.erase();
}

bool MatchRecvForAsyncRewrite(mlir::stablehlo::RecvOp op) {
    if (op.getIsHostTransfer()) return false;
    if (op.getNumResults() != 2) {  // one data result + one token result only.
        LOG(WARNING) << "MatchRecvForAsyncRewrite: multi-result stablehlo.recv shares one token "
                        "across all results, which xampi does not support splitting for async "
                        "operations, so will use synchronous communication.";
        return false;
    }
    auto attr = op.getSourceTargetPairsAttr();
    if (!attr) {
        LOG(WARNING) << "MatchRecvForAsyncRewrite: source_target_pairs is not set, so will use "
                        "synchronous communication.";
        return false;
    }
    return IsPlainDenseGroups(attr, "MatchRecvForAsyncRewrite");
}

void RewriteAsAsyncRecv(mlir::stablehlo::RecvOp op, const ProgramInfo& program_info) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext* context = op.getContext();
    mlir::Location loc = op.getLoc();

    mlir::Value incoming_token = op.getToken();
    mlir::Type result_type = op.getResult(0).getType();
    mlir::Type token_type = op.getResult(1).getType();

    auto api_version = mlir::stablehlo::CustomCallApiVersionAttr::get(
        context, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);
    int64_t channel_id = GetChannelId(op.getChannelHandle());
    ProcessGroupStrategy strategy = ResolvePermuteFamilyStrategy(channel_id);
    mlir::NamedAttribute request_buffer = MakeRequestBufferAttr(builder);

    llvm::SmallVector<mlir::NamedAttribute, 8> config_attrs = {
        builder.getNamedAttr("channel_id", builder.getI64IntegerAttr(channel_id)),
        request_buffer,
    };
    if (!AppendProcessGroupAttrs(builder, op.getSourceTargetPairsAttr(), strategy, program_info,
                                /*include_group_metadata=*/false, /*include_my_rank=*/true, config_attrs)) {
        LOG(FATAL) << "MatchRecvForAsyncRewrite verified source_target_pairs is "
                      "DenseIntElementsAttr. But AppendProcessGroupAttrs failed to match it as "
                      "such.";
    }
    auto backend_config = builder.getDictionaryAttr(config_attrs);

    llvm::SmallVector<mlir::NamedAttribute, 4> start_attrs = {
        builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.recv_start")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("backend_config", backend_config),
    };
    auto start_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type, token_type}, mlir::ValueRange{incoming_token}, start_attrs);

    auto alias = mlir::stablehlo::OutputOperandAliasAttr::get(
        context, /*outputTupleIndices=*/{0}, /*operandIndex=*/0, /*operandTupleIndices=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 5> done_attrs = {
        builder.getNamedAttr("call_target_name", builder.getStringAttr("xla_mpi.recv_done")),
        builder.getNamedAttr("has_side_effect", builder.getBoolAttr(true)),
        builder.getNamedAttr("mhlo.sharding", builder.getStringAttr("{manual}")),
        builder.getNamedAttr("api_version", api_version),
        builder.getNamedAttr("output_operand_aliases", builder.getArrayAttr({alias})),
        builder.getNamedAttr("backend_config", builder.getDictionaryAttr({request_buffer})),
    };
    auto done_op = builder.create<mlir::stablehlo::CustomCallOp>(
        loc, mlir::TypeRange{result_type, token_type},
        mlir::ValueRange{start_op.getResult(0), start_op.getResult(1)}, done_attrs);

    op.getResult(0).replaceAllUsesWith(done_op.getResult(0));
    op.getResult(1).replaceAllUsesWith(done_op.getResult(1));
    op.erase();
}

}  // namespace

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

    llvm::SmallVector<mlir::stablehlo::AllToAllOp> all_to_all_rewrites;
    entry.walk([&](mlir::stablehlo::AllToAllOp op) {
        if (MatchAllToAllForAsyncRewrite(op)) all_to_all_rewrites.push_back(op);
    });
    for (mlir::stablehlo::AllToAllOp op : all_to_all_rewrites) RewriteAsAsyncAllToAll(op, program_info);

    llvm::SmallVector<mlir::stablehlo::SendOp> send_rewrites;
    entry.walk([&](mlir::stablehlo::SendOp op) {
        if (MatchSendForAsyncRewrite(op)) send_rewrites.push_back(op);
    });
    for (mlir::stablehlo::SendOp op : send_rewrites) RewriteAsAsyncSend(op, program_info);

    llvm::SmallVector<mlir::stablehlo::RecvOp> recv_rewrites;
    entry.walk([&](mlir::stablehlo::RecvOp op) {
        if (MatchRecvForAsyncRewrite(op)) recv_rewrites.push_back(op);
    });
    for (mlir::stablehlo::RecvOp op : recv_rewrites) RewriteAsAsyncRecv(op, program_info);
}

}  // namespace xla_mpi
