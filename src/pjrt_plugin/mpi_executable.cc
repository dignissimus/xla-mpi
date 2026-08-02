#include "pjrt_plugin/mpi_executable.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/computation_placer.h"
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

PJRT_Buffer_Type MlirElementTypeToPjrtType(mlir::Type type) {
    return pjrt::ConvertToPjRtBufferType(xla::ConvertMlirTypeToPrimitiveType(type));
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
                                                     size_t code_size) {
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

    xla::ExecutableBuildOptions build_options;
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

    xla::ExecutableRunOptions run_options;
    run_options.set_allocator(client_->backend().memory_allocator());
    absl::StatusOr<xla::ExecutionOutput> output = executable_->Run(std::move(arguments), run_options);
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
    args->num_replicas = 1;
    return nullptr;
}

PJRT_Error* MPI_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args) {
    args->num_partitions = 1;
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
    static const char* kDevice = "device";
    static thread_local std::vector<const char*> kinds;
    static thread_local std::vector<size_t> kind_sizes;
    size_t n = args->executable->executable->num_outputs();
    kinds.assign(n, kDevice);
    kind_sizes.assign(n, 6);
    args->num_outputs = n;
    args->memory_kinds = kinds.data();
    args->memory_kind_sizes = kind_sizes.data();
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
    // Identity of the underlying MpiExecutable is a stable-enough fingerprint
    // for this phase (no persistent compilation cache yet).
    static thread_local std::string fp;
    fp = std::to_string(reinterpret_cast<uintptr_t>(
        args->executable->executable->executable.get()));
    args->executable_fingerprint = fp.c_str();
    args->executable_fingerprint_size = fp.size();
    return nullptr;
}

PJRT_Error* MPI_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
    static thread_local std::vector<PJRT_Buffer_Type> types;
    types.clear();
    for (const xla_mpi::OutputInfo& info : args->executable->executable->output_info()) {
        types.push_back(static_cast<PJRT_Buffer_Type>(info.dtype));
    }
    args->output_types = types.data();
    args->num_output_types = types.size();
    return nullptr;
}

PJRT_Error* MPI_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    static thread_local std::vector<int64_t> dims;
    static thread_local std::vector<size_t> dim_sizes;
    dims.clear();
    dim_sizes.clear();
    for (const xla_mpi::OutputInfo& info : args->executable->executable->output_info()) {
        dim_sizes.push_back(info.shape.size());
        dims.insert(dims.end(), info.shape.begin(), info.shape.end());
    }
    args->num_outputs = dim_sizes.size();
    args->dims = dims.data();
    args->dim_sizes = dim_sizes.data();
    return nullptr;
}

PJRT_Error* MPI_Executable_GetCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    return MakeError("MPI_Executable_GetCompiledMemoryStats not yet implemented");
}

PJRT_Error* MPI_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    static thread_local std::string fp;
    fp = std::to_string(reinterpret_cast<uintptr_t>(args->executable->executable.get()));
    args->executable_fingerprint = fp.c_str();
    args->executable_fingerprint_size = fp.size();
    return nullptr;
}

PJRT_Error* MPI_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
    int device_id = 0;
    if (!args->executable->addressable_devices.empty()) {
        device_id = args->executable->addressable_devices[0]->mpi_rank;
    }

    xla::DeviceAssignment device_assignment(/*replica_count=*/1, /*computation_count=*/1);
    device_assignment(0, 0) = device_id;

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
