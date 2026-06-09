#ifndef XLA_MPI_PJRT_PLUGIN_H_
#define XLA_MPI_PJRT_PLUGIN_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Plugin API
PJRT_Error* MPI_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args);
PJRT_Error* MPI_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args);

#endif  // XLA_MPI_PJRT_PLUGIN_H_
