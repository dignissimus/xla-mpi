#include "pjrt_plugin/pjrt_memory.h"
#include "pjrt_plugin/pjrt_types.h"
#include <iostream>


// ============================================================================
// Memory API
// ============================================================================

PJRT_Error* MPI_Memory_Id(PJRT_Memory_Id_Args* args) {
    // TODO: Defaulting to 0?
    args->id = args->memory ? static_cast<PJRT_Memory_Impl*>(args->memory)->id : 0;
    return nullptr;
}

PJRT_Error* MPI_Memory_Kind(PJRT_Memory_Kind_Args* args) {
    static const char* kind_str = "device";
    args->kind = kind_str;
    args->kind_size = 6;
    return nullptr;
}

PJRT_Error* MPI_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
    static const char* str = "MPI_Memory";
    args->debug_string = str;
    args->debug_string_size = 10;
    return nullptr;
}

PJRT_Error* MPI_Memory_ToString(PJRT_Memory_ToString_Args* args) {
    static const char* str = "MPI_Memory()";
    args->to_string = str;
    args->to_string_size = 12;
    return nullptr;
}

PJRT_Error* MPI_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
    auto* mem = args->memory ? static_cast<PJRT_Memory_Impl*>(args->memory) : nullptr;
    if (mem && mem->device) {
        static PJRT_Device* dev_array[1];
        dev_array[0] = mem->device;
        args->devices = dev_array;
        args->num_devices = 1;
    } else {
        args->devices = nullptr;
        args->num_devices = 0;
    }
    return nullptr;
}
