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
        
        PJRT_Device* device = new PJRT_Device();
        device->device = new xla_mpi::MpiDevice(); 
        device->client = g_default_client;

        PJRT_DeviceDescription* desc = new PJRT_DeviceDescription();
        desc->device = device;
        device->description = desc;

        auto* mem = new PJRT_Memory();
        mem->device = device;
        mem->client = g_default_client;
        mem->id = rank;
        device->default_memory = mem;
        g_default_client->memories.push_back(mem);


        g_default_client->devices.push_back(device);
        g_default_client->device_descriptions.push_back(desc);

        g_default_client->topology = new PJRT_TopologyDescription();
        g_default_client->topology->client = g_default_client;
    });

    args->client = g_default_client;
    return nullptr; 
}

PJRT_Error* MPI_Client_Destroy(PJRT_Client_Destroy_Args* args) {
    return nullptr; 
}

PJRT_Error* MPI_Client_Devices(PJRT_Client_Devices_Args* args) {
    args->devices = g_default_client->devices.data();
    args->num_devices = g_default_client->devices.size();
    return nullptr;
}

PJRT_Error* MPI_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
    args->addressable_devices = g_default_client->devices.data();
    args->num_addressable_devices = g_default_client->devices.size();
    std::cerr << "sz: " << g_default_client->devices.size()<< std::endl;
    // return MakeError("MPI Client Addressable Memories not implemented.");
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
    return MakeError("MemoryStats not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

