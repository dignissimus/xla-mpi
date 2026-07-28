#include "pjrt_plugin/pjrt_error.h"
#include "pjrt_plugin/pjrt_types.h"

void MPI_Error_Destroy(PJRT_Error_Destroy_Args* args) {
    if (args->error && args->error->vtable && args->error->vtable->destroy) {
        args->error->vtable->destroy(args->error);
    }
}

void MPI_Error_Message(PJRT_Error_Message_Args* args) {
    if (args->error && args->error->vtable && args->error->vtable->message) {
        args->error->vtable->message(args->error, &args->message, &args->message_size);
    } else {
        args->message = nullptr;
        args->message_size = 0;
    }
}

PJRT_Error* MPI_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    args->code = (args->error && args->error->vtable && args->error->vtable->get_code)
                     ? args->error->vtable->get_code(args->error)
                     : PJRT_Error_Code_OK;
    return nullptr;
}

PJRT_Error* MPI_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args) {
    if (args->error && args->error->vtable && args->error->vtable->for_each_payload) {
        args->error->vtable->for_each_payload(args->error, args->visitor, args->user_arg);
    }
    return nullptr;
}
