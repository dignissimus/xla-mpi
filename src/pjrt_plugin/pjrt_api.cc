#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/pjrt_plugin.h"
#include "pjrt_plugin/pjrt_client.h"
#include "pjrt_plugin/pjrt_device.h"
#include "pjrt_plugin/pjrt_memory.h"
#include "pjrt_plugin/pjrt_topology.h"
#include "pjrt_plugin/pjrt_error.h"

#include <mpi.h>
#include <iostream>

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
    .PJRT_Client_DefaultDeviceAssignment = MPI_Client_DefaultDeviceAssignment,

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

    .PJRT_TopologyDescription_Create = MPI_TopologyDescription_Create,
    .PJRT_TopologyDescription_Destroy = MPI_TopologyDescription_Destroy,
    .PJRT_TopologyDescription_PlatformName = MPI_TopologyDescription_PlatformName,
    .PJRT_TopologyDescription_PlatformVersion = MPI_TopologyDescription_PlatformVersion,
    .PJRT_TopologyDescription_GetDeviceDescriptions = MPI_TopologyDescription_GetDeviceDescriptions,
    .PJRT_TopologyDescription_Serialize = nullptr,
    .PJRT_TopologyDescription_Attributes = MPI_TopologyDescription_Attributes,

    .PJRT_Compile = MPI_Compile,

    .PJRT_Client_TopologyDescription = MPI_Client_TopologyDescription,
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
