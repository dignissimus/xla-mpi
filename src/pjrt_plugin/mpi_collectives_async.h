#ifndef XLA_MPI_MPI_COLLECTIVES_ASYNC_H_
#define XLA_MPI_MPI_COLLECTIVES_ASYNC_H_

#include <cstdint>
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

class MpiReduceScatterRequestBuffer {
public:
    void Store(MPI_Request request, void* scratch, std::vector<int64_t> dims,
              int64_t scatter_dimension, int64_t chunk_index, MPI_Datatype elem_type,
              MPI_Comm comm) {
        request_ = request;
        scratch_ = scratch;
        dims_ = std::move(dims);
        scatter_dimension_ = scatter_dimension;
        chunk_index_ = chunk_index;
        elem_type_ = elem_type;
        comm_ = comm;
    }

    absl::Status MpiWait(::xla::ffi::AnyBuffer recv_buffer,
                        ::xla::ffi::Result<::xla::ffi::AnyBuffer> result) {
        absl::Status status = MpiErrorToAbslStatus(MPI_Wait(&request_, MPI_STATUS_IGNORE));
        if (comm_ != MPI_COMM_WORLD) MPI_Comm_free(&comm_);
        if (!status.ok()) {
            ::operator delete(scratch_);
            return status;
        }

        int ndims = static_cast<int>(dims_.size());
        std::vector<int> full_sizes(dims_.begin(), dims_.end());
        std::vector<int> chunk_sizes = full_sizes;
        auto chunk_dim_size = static_cast<int>(recv_buffer.dimensions()[scatter_dimension_]);
        chunk_sizes[scatter_dimension_] = chunk_dim_size;
        std::vector<int> starts(ndims, 0);
        starts[scatter_dimension_] = static_cast<int>(chunk_index_) * chunk_dim_size;

        MPI_Datatype chunk_type;
        MPI_Type_create_subarray(ndims, full_sizes.data(), chunk_sizes.data(), starts.data(),
                                 MPI_ORDER_C, elem_type_, &chunk_type);
        MPI_Type_commit(&chunk_type);

        int position = 0;
        auto outsize = static_cast<int>(recv_buffer.size_bytes());
        absl::Status pack_status = MpiErrorToAbslStatus(MPI_Pack(
            scratch_, 1, chunk_type, result->untyped_data(), outsize, &position, MPI_COMM_SELF));

        MPI_Type_free(&chunk_type);
        ::operator delete(scratch_);
        return pack_status;
    }

private:
    MPI_Request request_;
    void* scratch_ = nullptr;
    std::vector<int64_t> dims_;
    int64_t scatter_dimension_ = 0;
    int64_t chunk_index_ = 0;
    MPI_Datatype elem_type_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

}  // namespace xla_mpi

#endif  // XLA_MPI_MPI_COLLECTIVES_ASYNC_H_
