#include "pjrt_plugin/mpi_executable.h"

#include <string>

#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"

namespace xla_mpi {

std::shared_ptr<MpiExecutable> MpiExecutable::Create(const std::string& format, const char* code,
                                                     size_t code_size) {
    std::shared_ptr<MpiExecutable> exe(new MpiExecutable());
    exe->error_ = "MpiExecutable::Create not yet implemented";
    return exe;
}

}  // namespace xla_mpi

PJRT_Error* MPI_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
    return MakeError("MPI_Executable_Destroy not yet implemented");
}
PJRT_Error* MPI_Executable_Name(PJRT_Executable_Name_Args* args) {
    return MakeError("MPI_Executable_Name not yet implemented");
}
PJRT_Error* MPI_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args) {
    return MakeError("MPI_Executable_NumReplicas not yet implemented");
}
PJRT_Error* MPI_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args) {
    return MakeError("MPI_Executable_NumPartitions not yet implemented");
}
PJRT_Error* MPI_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
    return MakeError("MPI_Executable_NumOutputs not yet implemented");
}
PJRT_Error* MPI_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    return MakeError("MPI_Executable_SizeOfGeneratedCodeInBytes not yet implemented");
}
PJRT_Error* MPI_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    return MakeError("MPI_Executable_GetCostAnalysis not yet implemented");
}
PJRT_Error* MPI_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    return MakeError("MPI_Executable_OutputMemoryKinds not yet implemented");
}
PJRT_Error* MPI_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args) {
    return MakeError("MPI_Executable_OptimizedProgram not yet implemented");
}
PJRT_Error* MPI_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
    return MakeError("MPI_Executable_Serialize not yet implemented");
}

PJRT_Error* MPI_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    return MakeError("MPI_LoadedExecutable_Destroy not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    return MakeError("MPI_LoadedExecutable_GetExecutable not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
    return MakeError("MPI_LoadedExecutable_AddressableDevices not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args) {
    return MakeError("MPI_LoadedExecutable_Delete not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
    return MakeError("MPI_LoadedExecutable_IsDeleted not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    return MakeError("MPI_LoadedExecutable_Execute not yet implemented");
}
PJRT_Error* MPI_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
    return MakeError("MPI_Executable_DeserializeAndLoad not yet implemented");
}
PJRT_Error* MPI_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
    return MakeError("MPI_LoadedExecutable_Fingerprint not yet implemented");
}

PJRT_Error* MPI_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
    return MakeError("MPI_Executable_OutputElementTypes not yet implemented");
}
PJRT_Error* MPI_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    return MakeError("MPI_Executable_OutputDimensions not yet implemented");
}

PJRT_Error* MPI_Executable_GetCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    return MakeError("MPI_Executable_GetCompiledMemoryStats not yet implemented");
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

PJRT_Error* MPI_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    return MakeError("MPI_Executable_Fingerprint not yet implemented");
}
