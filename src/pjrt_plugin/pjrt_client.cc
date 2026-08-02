#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/pjrt_types.h"
#include "pjrt_plugin/pjrt_mutex.h"
#include "pjrt_plugin/mpi_executable.h"

#include <mpi.h>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>

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
                // Can I nullptr this?
                PJRT_Memory* mem = MakeMemory(device, g_default_client, i);
                // TODO: Need to destroy mem
                device->default_memory = mem;
               device->client = g_default_client;

            if (i == rank) {
                                g_default_client->addressable_devices.push_back(device);
                                g_default_client->memories.push_back(mem);
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
            delete static_cast<PJRT_Memory_Impl*>(mem);
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

PJRT_Error* MPI_Client_UpdateGlobalProcessInfo(PJRT_Client_UpdateGlobalProcessInfo_Args* args) {
    return nullptr;
}

PJRT_Error* MPI_Client_Compile(PJRT_Client_Compile_Args* args) {
    std::scoped_lock lock(GetPjrtGlobalMutex());
    std::cout << "Compiling StableHLO program" << std::endl;

    PJRT_Client* client = GetClient(args->client);
    if (!client || !client->client) {
        // TODO: Why might this be the case
        // TODO: Think about wording. Is client appropriate here
        return MakeError("No MPI client");
    }
    // TODO: Any additional checks to make sure the client is ok? 
    // Maybe check devices or memories
    // TODO: Come back to this after handling failed mpi init
    
    // TODO: debug, remove later
    std::string format_str(args->program->format, args->program->format_size);
    std::cout << " Program format: " <<  format_str <<
                  " size " << args->program->format_size << std::endl;
    std::cout << " Program code size: %zu\n" << args->program->code_size << std::endl;

    std::shared_ptr<xla_mpi::MpiExecutable> mpi_executable =
        xla_mpi::MpiExecutable::Create(format_str, args->program->code, args->program->code_size);
    if (!mpi_executable->IsValid()) {
        return MakeError("Failed to compile StableHLO program: " + mpi_executable->error());
    }

    PJRT_Executable* executable = new PJRT_Executable();
    executable->executable = std::move(mpi_executable);
    executable->client = client;
    executable->owned_by_loaded = true;

    PJRT_LoadedExecutable* loaded = new PJRT_LoadedExecutable();
    loaded->executable = executable;
    loaded->client = client;
    loaded->addressable_devices = client->addressable_devices;

    args->executable = loaded;
    return nullptr;
}

PJRT_Error* MPI_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    PJRT_Client* client = GetClient(args->client);
    if (!client) {
        return MakeError("MPI_Client_BufferFromHostBuffer: no MPI client");
    }

    std::vector<int64_t> shape(args->dims, args->dims + args->num_dims);
    std::unique_ptr<xla_mpi::MpiBuffer> buffer;
    try {
        buffer = xla_mpi::MpiBuffer::CreateFromHost(args->data, args->type, shape);
    } catch (const std::exception& e) {
        return MakeError(std::string("MPI_Client_BufferFromHostBuffer: ") + e.what());
    }

    PJRT_Device* device = args->device;
    if (!device && args->memory) {
        device = static_cast<PJRT_Memory_Impl*>(args->memory)->device;
    }

    PJRT_Buffer* pjrt_buffer = new PJRT_Buffer();
    pjrt_buffer->buffer = std::move(buffer);
    pjrt_buffer->client = client;
    pjrt_buffer->device = device;
    pjrt_buffer->memory = args->memory ? args->memory
                                        : (device ? device->default_memory : nullptr);

    args->buffer = pjrt_buffer;
    args->done_with_host_buffer = new PJRT_Event{.ready = true};
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
    if (target_id >= 0 && target_id < args->client->devices.size()) {
        // TODO: I should check if the device is actually addressable
        // Then return MakeError if not
        args->addressable_device = args->client->devices[target_id];
        return nullptr;
    }

    // TODO: More informative
    return MakeError("Device with specified ID not found");
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
