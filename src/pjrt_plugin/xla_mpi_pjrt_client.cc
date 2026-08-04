#include "pjrt_plugin/xla_mpi_pjrt_client.h"
#include "pjrt_plugin/mlir_rewrite.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <mpi.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace xla_mpi {

namespace {

mlir::func::FuncOp FindEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;
    for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
        if (func_op.getName() == "main") return func_op;
        if (!entry) entry = func_op;
    }
    return entry;
}

ProgramInfo BuildProgramInfo(const xla::ExecutableBuildOptions& build_options) {
    ProgramInfo info;
    info.num_replicas = build_options.num_replicas();
    info.num_partitions = build_options.num_partitions();

    xla::DeviceAssignment device_assignment;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (build_options.has_device_assignment()) {
        device_assignment = build_options.device_assignment();
    } else {
        info.num_replicas = 1;
        info.num_partitions = 1;
        device_assignment = xla::DeviceAssignment(1, 1);
        device_assignment(0, 0) = PackCpuDeviceId(rank);
    }

    info.device_assignment_flat.reserve(
        static_cast<size_t>(info.num_replicas * info.num_partitions));
    for (int64_t r = 0; r < info.num_replicas; ++r) {
        for (int64_t p = 0; p < info.num_partitions; ++p) {
            info.device_assignment_flat.push_back(device_assignment(static_cast<int>(r),
                                                                     static_cast<int>(p)));
        }
    }

    // Resolve this rank's own (replica, partition) once here, at compile
    // time, rather than by every collective's FFI handler re-deriving it
    // from scratch on every runtime invocation of the compiled program (see
    // mpi_process_group.h's DeviceAt for the packed-id unpacking this
    // mirrors).
    info.my_rank = rank;
    info.my_replica = -1;
    info.my_partition = -1;
    for (int64_t r = 0; r < info.num_replicas && info.my_replica < 0; ++r) {
        for (int64_t p = 0; p < info.num_partitions; ++p) {
            int64_t idx = r * info.num_partitions + p;
            if (UnpackCpuProcessIndex(info.device_assignment_flat[idx]) == rank) {
                info.my_replica = r;
                info.my_partition = p;
                break;
            }
        }
    }
    return info;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> XlaMpiPjRtClient::Compile(
    xla::MaybeOwningMlirModule module, xla::CompileOptions options) {
    mlir::ModuleOp mlir_module = module.mlir_module();
    mlir::func::FuncOp entry = FindEntryFunction(mlir_module);
    if (entry && entry.getBody().hasOneBlock()) {
        RewriteCollectivesAsAsync(entry, BuildProgramInfo(options.executable_build_options));
    }
    return wrapped_->Compile(std::move(module), std::move(options));
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> XlaMpiPjRtClient::CompileAndLoad(
    xla::MaybeOwningMlirModule module, xla::CompileOptions options) {
    mlir::ModuleOp mlir_module = module.mlir_module();
    mlir::func::FuncOp entry = FindEntryFunction(mlir_module);
    if (entry && entry.getBody().hasOneBlock()) {
        RewriteCollectivesAsAsync(entry, BuildProgramInfo(options.executable_build_options));
    }
    return wrapped_->CompileAndLoad(std::move(module), std::move(options));
}

}  // namespace xla_mpi
