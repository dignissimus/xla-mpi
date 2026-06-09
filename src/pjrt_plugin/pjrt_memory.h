#ifndef XLA_MPI_PJRT_MEMORY_H_
#define XLA_MPI_PJRT_MEMORY_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Memory API
PJRT_Error* MPI_Memory_Id(PJRT_Memory_Id_Args* args);
PJRT_Error* MPI_Memory_Kind(PJRT_Memory_Kind_Args* args);
PJRT_Error* MPI_Memory_DebugString(PJRT_Memory_DebugString_Args* args);
PJRT_Error* MPI_Memory_ToString(PJRT_Memory_ToString_Args* args);
PJRT_Error* MPI_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args);

#endif  // XLA_MPI_PJRT_MEMORY_H_
