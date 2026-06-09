#include "pjrt_plugin/pjrt_error.h"
#include "pjrt_plugin/pjrt_types.h"

void MPI_Error_Destroy(PJRT_Error_Destroy_Args* args) {
    delete args->error;
}

void MPI_Error_Message(PJRT_Error_Message_Args* args) {
    if (args->error) {
        args->message = args->error->message.c_str();
        args->message_size = args->error->message.size();
    }
}

// The headerfile doesn't expose error code 0 as an enum
// The headerfile in later hashes do
// https://github.com/openxla/xla/blob/0d1b60216ea13b0d261d59552a0f7ef20c4f76c5/xla/pjrt/c/pjrt_c_api.h#L163
constexpr PJRT_Error_Code PJRT_Error_Code_OK = static_cast<PJRT_Error_Code>(0);
PJRT_Error* MPI_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    args->code = args->error ? args->error->code : PJRT_Error_Code_OK;
    return nullptr;
}
