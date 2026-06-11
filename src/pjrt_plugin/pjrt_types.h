#pragma once

#include <xla/pjrt/c/pjrt_c_api.h>
#include "pjrt_plugin/mpi_client.h"

#include <memory>
#include <mutex>
#include <string>
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
struct PJRT_Memory;

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

struct PJRT_Memory {
    PJRT_Device* device = nullptr;
    PJRT_Client* client = nullptr;
    int id = 0;
};

struct PJRT_TopologyDescription {
    PJRT_Client* client = nullptr;
};

struct PJRT_Buffer {
    std::unique_ptr<xla_mpi::MpiBuffer> buffer;
    PJRT_Client* client = nullptr;
};

struct PJRT_Executable {
    std::unique_ptr<xla_mpi::MpiExecutable> executable;
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

struct PJRT_Error {
    std::string message;
    PJRT_Error_Code code;
};

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL);

PJRT_Client* GetOrCreateDefaultClient();
PJRT_Client* GetClient(PJRT_Client* client);

extern const char* const kPlatformName;
extern const char* const kPlatformVersion;
