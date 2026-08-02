#ifndef XLA_MPI_MPI_EXECUTABLE_H_
#define XLA_MPI_MPI_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mpi_buffer.h"

namespace xla_mpi {

struct OutputInfo {
    int dtype;
    std::vector<int64_t> shape;
};

struct MpiExecuteResult {
    std::vector<MpiBuffer*> buffers;
    std::string error_message;
};

class MpiExecutable {
public:
    static std::shared_ptr<MpiExecutable> Create(const std::string& format, const char* code, size_t code_size);
    ~MpiExecutable() = default;

    bool IsValid() const { return valid_; }
    std::string error() const { return error_; }
    size_t num_outputs() const { return output_info_.size(); }
    const std::vector<OutputInfo>& output_info() const { return output_info_; }

    MpiExecuteResult Execute(const std::vector<MpiBuffer*>& inputs);

private:
    MpiExecutable() = default;

    bool valid_ = false;
    std::string error_;
    std::vector<OutputInfo> output_info_;
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

// PJRT_Error* MPI_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args);
// PJRT_Error* MPI_Executable_GetCompileOptions(PJRT_Executable_GetCompileOptions_Args* args);
PJRT_Error* MPI_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args);

#ifdef __cplusplus
}
#endif

#endif  // XLA_MPI_MPI_EXECUTABLE_H_
