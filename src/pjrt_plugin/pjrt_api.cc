#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_plugin/xla_mpi_pjrt_api.h"

extern "C" {

const PJRT_Api* GetPjrtApi() {
    return xla_mpi::GetXlaMpiPjrtApi();
}

}
