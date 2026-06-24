#ifndef XLA_MPI_PJRT_CLIENT_H_
#define XLA_MPI_PJRT_CLIENT_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Client API
PJRT_Error* MPI_Client_Create(PJRT_Client_Create_Args* args);
PJRT_Error* MPI_Client_Destroy(PJRT_Client_Destroy_Args* args);
PJRT_Error* MPI_Client_Devices(PJRT_Client_Devices_Args* args);
PJRT_Error* MPI_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args);
PJRT_Error* MPI_Client_PlatformName(PJRT_Client_PlatformName_Args* args);
PJRT_Error* MPI_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args);
PJRT_Error* MPI_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args);
PJRT_Error* MPI_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args);
PJRT_Error* MPI_Client_Compile(PJRT_Client_Compile_Args* args);
PJRT_Error* MPI_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args);


PJRT_Error* MPI_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args);
PJRT_Error* MPI_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args);

PJRT_Error* MPI_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args);
PJRT_Error* MPI_Compile(PJRT_Compile_Args* args);

#endif  // XLA_MPI_PJRT_CLIENT_H_
