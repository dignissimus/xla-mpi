#ifndef XLA_MPI_PJRT_DEVICE_H_
#define XLA_MPI_PJRT_DEVICE_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Device Description API
PJRT_Error* MPI_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args);
PJRT_Error* MPI_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args);
PJRT_Error* MPI_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args);
PJRT_Error* MPI_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args);
PJRT_Error* MPI_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args);
PJRT_Error* MPI_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args);

// Device API
PJRT_Error* MPI_Device_GetDescription(PJRT_Device_GetDescription_Args* args);
PJRT_Error* MPI_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args);
PJRT_Error* MPI_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args);
PJRT_Error* MPI_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args);
PJRT_Error* MPI_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args);
PJRT_Error* MPI_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args);

#endif  // XLA_MPI_PJRT_DEVICE_H_
