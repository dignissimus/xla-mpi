#pragma once

#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/mpi_client.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xla_mpi {
class MpiClient;
class MpiDevice;
class MpiBuffer;
class MpiExecutable;
} // namespace xla_mpi

struct PJRT_TopologyDescription;
struct PJRT_DeviceDescription;
struct PJRT_Device;
struct PJRT_Client;

// ---- PJRT_Error --------------------------------------------------------

struct PJRT_Error_Impl : public PJRT_Error {
    std::string message;
    PJRT_Error_Code code = PJRT_Error_Code_INTERNAL;

    static void Destroy(PJRT_Error* error);
    static void Message(const PJRT_Error* error, const char** message,
                         size_t* message_size);
    static PJRT_Error_Code GetCode(const PJRT_Error* error);
    static void ForEachPayload(const PJRT_Error* error,
                                PJRT_Error_PayloadVisitor visitor,
                                void* user_arg);
};

extern const PJRT_Error_FunctionTable kMpiErrorVtable;

// ---- PJRT_Memory --------------------------------------------------------

struct PJRT_Memory_Impl : public PJRT_Memory {
    PJRT_Device* device = nullptr;
    PJRT_Client* client = nullptr;
    int id = 0;

    ~PJRT_Memory_Impl();

    static void* GetUserData(PJRT_Memory* memory, const void* key);
    static void SetUserData(PJRT_Memory* memory, const void* key, void* data,
                             void (*dtor)(void*));

  private:
    struct UserDataEntry {
        void* data;
        void (*dtor)(void*);
    };
    std::unordered_map<const void*, UserDataEntry> user_data_;
};

extern const PJRT_Memory_FunctionTable kMpiMemoryVtable;

PJRT_Memory* MakeMemory(PJRT_Device* device, PJRT_Client* client, int id);

struct PJRT_Client {
    std::unique_ptr<xla_mpi::MpiClient> client;
    std::vector<PJRT_Device*> devices;
    std::vector<PJRT_Device*> addressable_devices;
    std::vector<PJRT_DeviceDescription*> device_descriptions;
    std::vector<PJRT_Memory*> memories;
    PJRT_TopologyDescription* topology = nullptr;
    int mpi_rank{};
};

struct PJRT_Device {
    xla_mpi::MpiDevice* device = nullptr;
    PJRT_Client* client = nullptr;
    PJRT_DeviceDescription* description = nullptr;
    PJRT_Memory* default_memory = nullptr;
    int mpi_rank{};
};

struct PJRT_DeviceDescription {
    PJRT_Device* device = nullptr;
    int mpi_rank{};
    std::string debug_string;
};

struct PJRT_TopologyDescription {
    PJRT_Client* client = nullptr;
};

struct PJRT_Buffer {
    std::unique_ptr<xla_mpi::MpiBuffer> buffer;
    PJRT_Client* client = nullptr;
    PJRT_Device* device = nullptr;
    PJRT_Memory* memory = nullptr;
    bool deleted = false;
};

struct PJRT_Executable {
    std::shared_ptr<xla_mpi::MpiExecutable> executable;
    PJRT_Client* client = nullptr;
    bool owned_by_loaded = false;
};

struct PJRT_LoadedExecutable {
    PJRT_Executable* executable = nullptr;
    PJRT_Client* client = nullptr;
    std::vector<PJRT_Device*> addressable_devices;
};

struct PJRT_Event {
    bool ready = true;
};

PJRT_Error* MPI_Event_Destroy(PJRT_Event_Destroy_Args* args);
PJRT_Error* MPI_Event_IsReady(PJRT_Event_IsReady_Args* args);
PJRT_Error* MPI_Event_Error(PJRT_Event_Error_Args* args);
PJRT_Error* MPI_Event_Await(PJRT_Event_Await_Args* args);
PJRT_Error* MPI_Event_OnReady(PJRT_Event_OnReady_Args* args);

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL);

PJRT_Client* GetOrCreateDefaultClient();
PJRT_Client* GetClient(PJRT_Client* client);

extern const char* const kPlatformName;
extern const char* const kPlatformVersion;
