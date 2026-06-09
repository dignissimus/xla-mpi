#ifndef XLA_MPI_PJRT_ERROR_H_
#define XLA_MPI_PJRT_ERROR_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Error API
void MPI_Error_Destroy(PJRT_Error_Destroy_Args* args);
void MPI_Error_Message(PJRT_Error_Message_Args* args);
PJRT_Error* MPI_Error_GetCode(PJRT_Error_GetCode_Args* args);

#endif  // XLA_MPI_PJRT_ERROR_H_
