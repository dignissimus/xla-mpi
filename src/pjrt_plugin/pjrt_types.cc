#include "pjrt_plugin/pjrt_types.h"

const char* const kPlatformName = "mpi";
const char* const kPlatformVersion = "0.1.0";

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code) {
    PJRT_Error* error = new PJRT_Error();
    error->message = msg;
    error->code = code;
    return error;
}

PJRT_Client* GetClient(PJRT_Client* client) {
    return client ? client : GetOrCreateDefaultClient();
}
