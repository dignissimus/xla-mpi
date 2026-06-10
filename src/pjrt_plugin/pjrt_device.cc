#include "pjrt_plugin/pjrt_device.h"
#include "pjrt_plugin/pjrt_error.h"
#include "pjrt_plugin/pjrt_types.h"
#include "pjrt_plugin/mpi_client.h"
#include <iostream>

#include <string>

PJRT_Error* MPI_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
    if (args->device_description && args->device_description->device) {
        args->id = args->device_description->device->device
                       ? args->device_description->device->device->id()
                       : 0;
    } else {
        args->id = 0;
    }
    return nullptr;
}

PJRT_Error* MPI_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args) {
    // TODO: Check assumptions
    // TODO: Process index into what?
    return 0;
    if (args->device_description && args->device_description->device) {
        args->process_index = args->device_description->device->device
                                  ? args->device_description->device->device->id()
                                  : 0;
        std::cerr << "If branch" <<  args->process_index <<std::endl;
    } else {
        args->process_index = 0;
        std::cerr << "Else branch" << args->process_index << std::endl;
    }
    return nullptr;
}

PJRT_Error* MPI_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

PJRT_Error* MPI_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
    static const char* kind = "mpi";
    args->device_kind = kind;
    args->device_kind_size = 3;
    return nullptr;
}

PJRT_Error* MPI_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args) {
    const std::string& debug_string = args->device_description->debug_string;
    args->debug_string = debug_string.data();
    args->debug_string_size = debug_string.size();
    return nullptr;
}

PJRT_Error* MPI_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args) {
    static const char* str = "MpiDevice()";
    args->to_string = str;
    args->to_string_size = 11;
    return nullptr;
}

PJRT_Error* MPI_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
    if (args->device) {
        args->device_description = args->device->description;
    } else {
        args->device_description = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPI_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
    args->is_addressable = true;
    return nullptr;
}

PJRT_Error* MPI_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
    if (args->device && args->device->device) {
        args->local_hardware_id = args->device->device->id();
    } else {
        args->local_hardware_id = 0;
    }
    return nullptr;
}

PJRT_Error* MPI_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
    if (args->device && args->device->default_memory) {
        static PJRT_Memory* mem_array[1];
        mem_array[0] = args->device->default_memory;
        args->memories = mem_array;
        args->num_memories = 1;
    } else {
        // TODO: Might want to assert that we're not in this branch
        // XLA might have CHECK macro?
        args->memories = nullptr;
        args->num_memories = 0;
    }
    return nullptr;
}

PJRT_Error* MPI_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
    if (args->device) {
        args->memory = args->device->default_memory;
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPI_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
    return MakeError("MemoryStats not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}
