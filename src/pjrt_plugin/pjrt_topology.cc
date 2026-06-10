#include "pjrt_plugin/pjrt_topology.h"
#include "pjrt_plugin/pjrt_types.h"

PJRT_Error* MPI_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args) {
    if (args->client) {
        args->topology = args->client->topology;
    } else {
        args->topology = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
    args->platform_name = "mpi";
    args->platform_name_size = 3;
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args) {
    args->platform_version = "0.1.0";
    args->platform_version_size = 5;
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
    if (args->topology && args->topology->client) {
        args->descriptions = args->topology->client->device_descriptions.data();
        args->num_descriptions = args->topology->client->device_descriptions.size();
    } else {
        args->descriptions = nullptr;
        args->num_descriptions = 0;
    }
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args) {
    // TODO: Memory ownership
    return nullptr;
}


PJRT_Error* MPI_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args* args) {
    args->topology = nullptr;
    return nullptr;
}

PJRT_Error* MPI_TopologyDescription_Serialize(PJRT_TopologyDescription_Serialize_Args* args) {
    return MakeError("TopologyDescription_Serialize not implemented",
                     PJRT_Error_Code_UNIMPLEMENTED);
}

