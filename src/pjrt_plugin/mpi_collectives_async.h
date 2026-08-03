#ifndef XLA_MPI_MPI_COLLECTIVES_ASYNC_H_
#define XLA_MPI_MPI_COLLECTIVES_ASYNC_H_

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "pjrt_plugin/mpi_collectives.h"
#include "xla/ffi/ffi.h"

namespace xla_mpi {

class MpiRequestBuffer {
public:
    void Store(std::vector<MPI_Request> requests, std::vector<MPI_Datatype> types = {},
              MPI_Comm comm = MPI_COMM_NULL) {
        requests_ = std::move(requests);
        types_ = std::move(types);
        comm_ = comm;
    }

    absl::Status MpiWait() {
        absl::Status status = MpiErrorToAbslStatus(MPI_Waitall(
            static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE));
        for (MPI_Datatype type : types_) MPI_Type_free(&type);
        if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD) MPI_Comm_free(&comm_);
        requests_.clear();
        types_.clear();
        comm_ = MPI_COMM_NULL;
        return status;
    }

private:
    std::vector<MPI_Request> requests_;
    std::vector<MPI_Datatype> types_;
    MPI_Comm comm_ = MPI_COMM_NULL;
};

}  // namespace xla_mpi

#endif  // XLA_MPI_MPI_COLLECTIVES_ASYNC_H_
