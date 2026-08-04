#ifndef XLA_MPI_XLA_MPI_PJRT_API_H_
#define XLA_MPI_XLA_MPI_PJRT_API_H_

#include "xla/pjrt/c/pjrt_c_api.h"

namespace xla_mpi {

const PJRT_Api* GetXlaMpiPjrtApi();

}  // namespace xla_mpi

#endif  // XLA_MPI_XLA_MPI_PJRT_API_H_
