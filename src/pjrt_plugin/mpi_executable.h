#ifndef XLA_MPI_MPI_EXECUTABLE_H_
#define XLA_MPI_MPI_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mpi_buffer.h"
#include "xla/client/local_client.h"
#include "xla/service/computation_placer.h"

namespace xla_mpi {

struct OutputInfo {
    int dtype;
    std::vector<int64_t> shape;
};

struct MpiExecuteResult {
    std::vector<MpiBuffer*> buffers;
    std::string error_message;
};

// Compiles a StableHLO program using XLA's CPU compiler.
class MpiExecutable {
public:
    static std::shared_ptr<MpiExecutable> Create(const std::string& format, const char* code, size_t code_size,
                                                 const char* compile_options, size_t compile_options_size);
    ~MpiExecutable();

    bool IsValid() const;
    std::string error() const;
    size_t num_outputs() const;

    const std::vector<PJRT_Buffer_Type>& output_types() const { return output_types_; }
    const std::vector<int64_t>& output_dims() const { return output_dims_; }
    const std::vector<size_t>& output_dim_sizes() const { return output_dim_sizes_; }
    const std::vector<const char*>& output_memory_kinds() const { return output_memory_kinds_; }
    const std::vector<size_t>& output_memory_kind_sizes() const { return output_memory_kind_sizes_; }
    const std::string& fingerprint() const { return fingerprint_; }

    int64_t num_replicas() const { return num_replicas_; }
    int64_t num_partitions() const { return num_partitions_; }
    const xla::DeviceAssignment& device_assignment() const { return device_assignment_; }

    MpiExecuteResult Execute(const std::vector<MpiBuffer*>& inputs);

private:
    struct ArgInfo {
        int dtype;
        std::vector<int64_t> shape;
    };

    MpiExecutable() = default;

    bool valid_ = false;
    std::string error_;
    std::vector<ArgInfo> input_info_;
    std::vector<OutputInfo> output_info_;
    std::vector<PJRT_Buffer_Type> output_types_;
    std::vector<int64_t> output_dims_;
    std::vector<size_t> output_dim_sizes_;
    std::vector<const char*> output_memory_kinds_;
    std::vector<size_t> output_memory_kind_sizes_;
    std::string fingerprint_;
    int64_t num_replicas_ = 1;
    int64_t num_partitions_ = 1;
    xla::DeviceAssignment device_assignment_;
    std::unique_ptr<xla::LocalExecutable> executable_;
    // Not owned: xla::ClientLibrary owns a process-wide singleton per
    // platform. Needed again at execution time
    xla::LocalClient* client_ = nullptr;
};

}  // namespace xla_mpi

#include "pjrt_plugin/pjrt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

PJRT_Error* MPI_Executable_Destroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* MPI_Executable_Name(PJRT_Executable_Name_Args* args);
PJRT_Error* MPI_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args);
PJRT_Error* MPI_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args);
PJRT_Error* MPI_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args);
PJRT_Error* MPI_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);
PJRT_Error* MPI_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args);
PJRT_Error* MPI_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args);
PJRT_Error* MPI_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args);
PJRT_Error* MPI_Executable_Serialize(PJRT_Executable_Serialize_Args* args);

PJRT_Error* MPI_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args);
PJRT_Error* MPI_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args);
PJRT_Error* MPI_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args);
PJRT_Error* MPI_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args);
PJRT_Error* MPI_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args);
PJRT_Error* MPI_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args);
PJRT_Error* MPI_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args);
PJRT_Error* MPI_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args);

PJRT_Error* MPI_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args);
PJRT_Error* MPI_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args);

PJRT_Error* MPI_Executable_GetCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args* args);

PJRT_Error* MPI_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args);
PJRT_Error* MPI_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args);

#ifdef __cplusplus
}
#endif

#endif  // XLA_MPI_MPI_EXECUTABLE_H_
