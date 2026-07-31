#include "pjrt_plugin/pjrt_types.h"

const char* const kPlatformName = "mpi";
const char* const kPlatformVersion = "0.1.0";

// ============================================================================
// PJRT_Error_Impl
// ============================================================================

void PJRT_Error_Impl::Destroy(PJRT_Error* error) {
    delete static_cast<PJRT_Error_Impl*>(error);
}

void PJRT_Error_Impl::Message(const PJRT_Error* error, const char** message,
                               size_t* message_size) {
    const auto* impl = static_cast<const PJRT_Error_Impl*>(error);
    *message = impl->message.c_str();
    *message_size = impl->message.size();
}

PJRT_Error_Code PJRT_Error_Impl::GetCode(const PJRT_Error* error) {
    return static_cast<const PJRT_Error_Impl*>(error)->code;
}

void PJRT_Error_Impl::ForEachPayload(const PJRT_Error* error,
                                      PJRT_Error_PayloadVisitor visitor,
                                      void* user_arg) {
}

const PJRT_Error_FunctionTable kMpiErrorVtable = {
    /*struct_size=*/PJRT_Error_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/PJRT_Error_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*destroy=*/PJRT_Error_Impl::Destroy,
    /*message=*/PJRT_Error_Impl::Message,
    /*get_code=*/PJRT_Error_Impl::GetCode,
    /*for_each_payload=*/PJRT_Error_Impl::ForEachPayload,
};

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code) {
    PJRT_Error_Impl* error = new PJRT_Error_Impl();
    error->vtable = &kMpiErrorVtable;
    error->message = msg;
    error->code = code;
    return error;
}

// ============================================================================
// PJRT_Memory_Impl
// ============================================================================

PJRT_Memory_Impl::~PJRT_Memory_Impl() {
    for (auto& [key, entry] : user_data_) {
        if (entry.dtor && entry.data) {
            entry.dtor(entry.data);
        }
    }
}

void* PJRT_Memory_Impl::GetUserData(PJRT_Memory* memory, const void* key) {
    auto* self = static_cast<PJRT_Memory_Impl*>(memory);
    auto it = self->user_data_.find(key);
    return it != self->user_data_.end() ? it->second.data : nullptr;
}

void PJRT_Memory_Impl::SetUserData(PJRT_Memory* memory, const void* key,
                                    void* data, void (*dtor)(void*)) {
    auto* self = static_cast<PJRT_Memory_Impl*>(memory);
    auto it = self->user_data_.find(key);
    if (it != self->user_data_.end()) {
        if (it->second.dtor && it->second.data) {
            it->second.dtor(it->second.data);
        }
    }
    if (data == nullptr) {
        self->user_data_.erase(key);
    } else {
        self->user_data_[key] = UserDataEntry{data, dtor};
    }
}

const PJRT_Memory_FunctionTable kMpiMemoryVtable = {
    /*struct_size=*/PJRT_Memory_FunctionTable_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*instance_struct_size=*/PJRT_Memory_STRUCT_SIZE,
    /*get_user_data=*/PJRT_Memory_Impl::GetUserData,
    /*set_user_data=*/PJRT_Memory_Impl::SetUserData,
};

PJRT_Memory* MakeMemory(PJRT_Device* device, PJRT_Client* client, int id) {
    PJRT_Memory_Impl* memory = new PJRT_Memory_Impl();
    memory->vtable = &kMpiMemoryVtable;
    memory->device = device;
    memory->client = client;
    memory->id = id;
    return memory;
}

PJRT_Client* GetClient(PJRT_Client* client) {
    return client ? client : GetOrCreateDefaultClient();
}

// ============================================================================
// PJRT_Event
// ============================================================================

PJRT_Error* MPI_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    delete args->event;
    return nullptr;
}

PJRT_Error* MPI_Event_IsReady(PJRT_Event_IsReady_Args* args) {
    args->is_ready = args->event ? args->event->ready : true;
    return nullptr;
}

PJRT_Error* MPI_Event_Error(PJRT_Event_Error_Args* args) {
    return nullptr;
}

PJRT_Error* MPI_Event_Await(PJRT_Event_Await_Args* args) {
    return nullptr;
}

PJRT_Error* MPI_Event_OnReady(PJRT_Event_OnReady_Args* args) {
    args->callback(nullptr, args->user_arg);
    return nullptr;
}
