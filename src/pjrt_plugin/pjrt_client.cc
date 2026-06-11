#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/pjrt_types.h"

#include <mpi.h>
#include <iostream>
#include <mutex>

PJRT_Error* MPI_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    std::cout << "MPI_Plugin_Initialize" << std::endl;
    return nullptr;
}

PJRT_Error* MPI_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

static PJRT_Client* g_default_client = nullptr;
static std::once_flag init_flag;

// TODO: jax-mps impl is in _api and has name GetOrCreateDefaultClient or along those lines
PJRT_Error* MPI_Client_Create(PJRT_Client_Create_Args* args) {
    std::call_once(init_flag, []() {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::cout << "MPI_Client_Create on Rank " << rank << std::endl;
        
        g_default_client = new PJRT_Client();
        g_default_client->client = std::make_unique<xla_mpi::MpiClient>();
        g_default_client->mpi_rank = rank;
        
        for (int i = 0; i < size; ++i) {
            PJRT_Device* device = new PJRT_Device();
            device->device = new xla_mpi::MpiDevice(i); 
            // device->client = g_default_client;
            device->mpi_rank = i;

            PJRT_DeviceDescription* description = new PJRT_DeviceDescription();
            description->device = device;
            description->debug_string = "mpi_rank_" + std::to_string(i);
            description->mpi_rank = i;
            device->description = description;

                // TODO: Unsure if non-addressable gets PJRT_Memory
                auto* mem = new PJRT_Memory();
                mem->device = device;
                mem->client = g_default_client;
                mem->id = i;
                device->default_memory = mem;
                g_default_client->memories.push_back(mem);

            if (i == rank) {
                   device->client = g_default_client;
                                g_default_client->addressable_devices.push_back(device);
            }

            g_default_client->devices.push_back(device);
            g_default_client->device_descriptions.push_back(description);
        }

        g_default_client->topology = new PJRT_TopologyDescription();
        g_default_client->topology->client = g_default_client;
    });

    args->client = g_default_client;
    return nullptr; 
}

PJRT_Error* MPI_Client_Destroy(PJRT_Client_Destroy_Args* args) {
    if (g_default_client != nullptr) {
        delete g_default_client->topology;

        for (auto* desc : g_default_client->device_descriptions) {
            delete desc;
        }

        for (auto* device : g_default_client->devices) {
            delete device;
        }

        for (auto* mem : g_default_client->memories) {
            delete mem;
        }

        delete g_default_client;
        g_default_client = nullptr;
    }

    // TODO: Check MPI_Finalized error code and return accordingly
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }

    return nullptr;
}

PJRT_Error* MPI_Client_Devices(PJRT_Client_Devices_Args* args) {
    args->devices = args->client->devices.data();
    args->num_devices = args->client->devices.size();
    std::cerr << "sz: " << args->client->devices.size()<< std::endl;
    return nullptr;
}

PJRT_Error* MPI_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
    args->addressable_devices = args->client->addressable_devices.data();
    args->num_addressable_devices = args->client->addressable_devices.size();

    /*
    std::cerr << "sz: " << g_default_client->devices.size()<< std::endl;
    // return MakeError("MPI Client Addressable Memories not implemented.");
    // */
    return nullptr;
}

PJRT_Error* MPI_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
    args->platform_name = "mpi";
    args->platform_name_size = 3;
    return nullptr;
}

PJRT_Error* MPI_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "Was this before init ; rank = " << rank << std::endl;
    args->process_index = rank;
    return nullptr;
}

PJRT_Error* MPI_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
    static const char* version = "0.1.0";
    args->platform_version = version;
    args->platform_version_size = 5;
    return nullptr;
}

PJRT_Error* MPI_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
    // TODO: Should assert size as 1
    args->num_addressable_memories = args->client->memories.size();
    args->addressable_memories = args->client->memories.data();
    return nullptr;
}


PJRT_Error* MPI_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
    int target_id = args->id;
    if (target_id >= 0 && target_id < args->client->devices.size()) {
        args->device = args->client->devices[target_id];
        return nullptr;
    }

    // TODO: More informative
    return MakeError("Device with specified ID not found");
}

PJRT_Error* MPI_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args) {
    int target_id = args->local_hardware_id;
    std::cerr << "Target local hardware id " << target_id << std::endl;
    if (target_id >= 0 && target_id < args->client->devices.size()) {
        args->addressable_device = args->client->devices[target_id];
        return nullptr;
    }

    // TODO: More informative
    return MakeError("Device with specified ID not found");

    /*int local_id = args->local_hardware_id;
    if (local_id >= 0 && local_id < g_default_client->addressable_devices.size()) {
        args->addressable_device = g_default_client->addressable_devices[local_id];
        return nullptr;
    }
    return MakeError("MPI Client: Local hardware ID not found.");*/
}


PJRT_Error* MPI_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args) {
    // TODO: not correct for multirank mpi
    // Simple single-device assignment
    if (args->default_assignment && args->default_assignment_size > 0) {
        args->default_assignment[0] = 0;
    }
    return nullptr;
}


PJRT_Error* MPI_Compile(PJRT_Compile_Args* args) {
    return MakeError("PJRT_Compile not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}
