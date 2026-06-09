#ifndef XLA_MPI_PJRT_TOPOLOGY_H_
#define XLA_MPI_PJRT_TOPOLOGY_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Topology API
PJRT_Error* MPI_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args);
PJRT_Error* MPI_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args);
PJRT_Error* MPI_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args);
PJRT_Error* MPI_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args);
PJRT_Error* MPI_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args);
PJRT_Error* MPI_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args);

#endif  // XLA_MPI_PJRT_TOPOLOGY_H_
