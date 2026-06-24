#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/pjrt_plugin.h"
#include "pjrt_plugin/pjrt_client.h"
#include "pjrt_plugin/pjrt_device.h"
#include "pjrt_plugin/pjrt_memory.h"
#include "pjrt_plugin/pjrt_topology.h"
#include "pjrt_plugin/mpi_buffer.h"
#include "pjrt_plugin/pjrt_error.h"
#include "pjrt_plugin/mpi_executable.h"

#include <mpi.h>
#include <iostream>

// static PJRT_Error* MPI_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args) {
//     return nullptr;
// }

static const PJRT_Api pjrt_api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .extension_start = nullptr,

    .pjrt_api_version =
        {
            .struct_size = PJRT_Api_Version_STRUCT_SIZE,
            .extension_start = nullptr,
            .major_version = PJRT_API_MAJOR,
            .minor_version = PJRT_API_MINOR,
        },

    .PJRT_Error_Destroy = MPI_Error_Destroy,
    .PJRT_Error_Message = MPI_Error_Message,
    .PJRT_Error_GetCode = MPI_Error_GetCode,

    .PJRT_Plugin_Initialize = MPI_Plugin_Initialize,
    .PJRT_Plugin_Attributes = MPI_Plugin_Attributes,
    
    .PJRT_Client_Create = MPI_Client_Create,
    .PJRT_Client_Destroy = MPI_Client_Destroy,
    .PJRT_Client_PlatformName = MPI_Client_PlatformName,
    .PJRT_Client_ProcessIndex = MPI_Client_ProcessIndex,
    .PJRT_Client_PlatformVersion = MPI_Client_PlatformVersion,
    .PJRT_Client_Devices = MPI_Client_Devices,
    .PJRT_Client_AddressableDevices = MPI_Client_AddressableDevices,
    .PJRT_Client_LookupDevice = MPI_Client_LookupDevice,
    .PJRT_Client_LookupAddressableDevice = MPI_Client_LookupAddressableDevice,
    .PJRT_Client_AddressableMemories = MPI_Client_AddressableMemories,
    .PJRT_Client_Compile = MPI_Client_Compile,
    .PJRT_Client_DefaultDeviceAssignment = MPI_Client_DefaultDeviceAssignment,
    .PJRT_Client_BufferFromHostBuffer = MPI_Client_BufferFromHostBuffer,

    .PJRT_DeviceDescription_Id = MPI_DeviceDescription_Id,
    .PJRT_DeviceDescription_ProcessIndex = MPI_DeviceDescription_ProcessIndex,
    .PJRT_DeviceDescription_Attributes = MPI_DeviceDescription_Attributes,
    .PJRT_DeviceDescription_Kind = MPI_DeviceDescription_Kind,
    .PJRT_DeviceDescription_DebugString = MPI_DeviceDescription_DebugString,
    .PJRT_DeviceDescription_ToString = MPI_DeviceDescription_ToString,

    .PJRT_Device_GetDescription = MPI_Device_GetDescription,
    .PJRT_Device_IsAddressable = MPI_Device_IsAddressable,
    .PJRT_Device_LocalHardwareId = MPI_Device_LocalHardwareId,
    .PJRT_Device_AddressableMemories = MPI_Device_AddressableMemories,
    .PJRT_Device_DefaultMemory = MPI_Device_DefaultMemory,
    .PJRT_Device_MemoryStats = MPI_Device_MemoryStats,

    .PJRT_Memory_Id = MPI_Memory_Id,
    .PJRT_Memory_Kind = MPI_Memory_Kind,
    .PJRT_Memory_DebugString = MPI_Memory_DebugString,
    .PJRT_Memory_ToString = MPI_Memory_ToString,
    .PJRT_Memory_AddressableByDevices = MPI_Memory_AddressableByDevices,

    .PJRT_Executable_Destroy = MPI_Executable_Destroy,
    .PJRT_Executable_Name = MPI_Executable_Name,
    .PJRT_Executable_NumReplicas = MPI_Executable_NumReplicas,
    .PJRT_Executable_NumPartitions = MPI_Executable_NumPartitions,
    .PJRT_Executable_NumOutputs = MPI_Executable_NumOutputs,
    .PJRT_Executable_SizeOfGeneratedCodeInBytes = MPI_Executable_SizeOfGeneratedCodeInBytes,
    .PJRT_Executable_GetCostAnalysis = MPI_Executable_GetCostAnalysis,
    .PJRT_Executable_OutputMemoryKinds = MPI_Executable_OutputMemoryKinds,
    .PJRT_Executable_OptimizedProgram = MPI_Executable_OptimizedProgram,
    .PJRT_Executable_Serialize = MPI_Executable_Serialize,

    .PJRT_LoadedExecutable_Destroy = MPI_LoadedExecutable_Destroy,
    .PJRT_LoadedExecutable_GetExecutable = MPI_LoadedExecutable_GetExecutable,
    .PJRT_LoadedExecutable_AddressableDevices = MPI_LoadedExecutable_AddressableDevices,
    .PJRT_LoadedExecutable_Delete = MPI_LoadedExecutable_Delete,
    .PJRT_LoadedExecutable_IsDeleted = MPI_LoadedExecutable_IsDeleted,
    .PJRT_LoadedExecutable_Execute = MPI_LoadedExecutable_Execute,
    .PJRT_Executable_DeserializeAndLoad = MPI_Executable_DeserializeAndLoad,
    .PJRT_LoadedExecutable_Fingerprint = MPI_LoadedExecutable_Fingerprint,


    .PJRT_Buffer_Destroy = MPI_Buffer_Destroy,
    .PJRT_Buffer_ElementType = MPI_Buffer_ElementType,
    .PJRT_Buffer_Dimensions = MPI_Buffer_Dimensions,
    .PJRT_Buffer_UnpaddedDimensions = MPI_Buffer_UnpaddedDimensions,
    .PJRT_Buffer_DynamicDimensionIndices = MPI_Buffer_DynamicDimensionIndices,
    .PJRT_Buffer_GetMemoryLayout = MPI_Buffer_GetMemoryLayout,
    .PJRT_Buffer_OnDeviceSizeInBytes = MPI_Buffer_OnDeviceSizeInBytes,
    .PJRT_Buffer_Device = MPI_Buffer_Device,
    .PJRT_Buffer_Memory = MPI_Buffer_Memory,
    .PJRT_Buffer_Delete = MPI_Buffer_Delete,
    .PJRT_Buffer_IsDeleted = MPI_Buffer_IsDeleted,
    .PJRT_Buffer_CopyToDevice = MPI_Buffer_CopyToDevice,
    .PJRT_Buffer_ToHostBuffer = MPI_Buffer_ToHostBuffer,
    .PJRT_Buffer_IsOnCpu = MPI_Buffer_IsOnCpu,
    .PJRT_Buffer_ReadyEvent = MPI_Buffer_ReadyEvent,
    .PJRT_Buffer_UnsafePointer = MPI_Buffer_UnsafePointer,
    .PJRT_Buffer_IncreaseExternalReferenceCount = MPI_Buffer_IncreaseExternalReferenceCount,
    .PJRT_Buffer_DecreaseExternalReferenceCount = MPI_Buffer_DecreaseExternalReferenceCount,
    .PJRT_Buffer_OpaqueDeviceMemoryDataPointer = MPI_Buffer_OpaqueDeviceMemoryDataPointer,

    .PJRT_CopyToDeviceStream_Destroy = MPI_CopyToDeviceStream_Destroy,
    .PJRT_CopyToDeviceStream_AddChunk = MPI_CopyToDeviceStream_AddChunk,
    .PJRT_CopyToDeviceStream_TotalBytes = MPI_CopyToDeviceStream_TotalBytes,
    .PJRT_CopyToDeviceStream_GranuleSize = MPI_CopyToDeviceStream_GranuleSize,
    .PJRT_CopyToDeviceStream_CurrentBytes = MPI_CopyToDeviceStream_CurrentBytes,



    .PJRT_TopologyDescription_Create = MPI_TopologyDescription_Create,
    .PJRT_TopologyDescription_Destroy = MPI_TopologyDescription_Destroy,
    .PJRT_TopologyDescription_PlatformName = MPI_TopologyDescription_PlatformName,
    .PJRT_TopologyDescription_PlatformVersion = MPI_TopologyDescription_PlatformVersion,
    .PJRT_TopologyDescription_GetDeviceDescriptions = MPI_TopologyDescription_GetDeviceDescriptions,
    .PJRT_TopologyDescription_Serialize = nullptr,
    .PJRT_TopologyDescription_Attributes = MPI_TopologyDescription_Attributes,

    .PJRT_Compile = MPI_Compile,

    .PJRT_Executable_OutputElementTypes = MPI_Executable_OutputElementTypes,
    .PJRT_Executable_OutputDimensions = MPI_Executable_OutputDimensions,
    .PJRT_Buffer_CopyToMemory = MPI_Buffer_CopyToMemory,
    .PJRT_Executable_Fingerprint = MPI_Executable_Fingerprint,

    .PJRT_Client_TopologyDescription = MPI_Client_TopologyDescription,

    .PJRT_Executable_GetCompiledMemoryStats = MPI_Executable_GetCompiledMemoryStats,

    // .PJRT_LoadedExecutable_GetDeviceAssignment = MPI_LoadedExecutable_GetDeviceAssignment,
    // .PJRT_Executable_GetCompileOptions = MPI_Executable_GetCompileOptions,
    // .PJRT_Error_ForEachPayload = MPI_Error_ForEachPayload,
    // .PJRT_TopologyDescription_Fingerprint = YYY_TopologyDescription_Fingerprint,



};

extern "C" {

const PJRT_Api* GetPjrtApi() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "MPI Rank " << rank << std::endl;


    return &pjrt_api;
}

}
