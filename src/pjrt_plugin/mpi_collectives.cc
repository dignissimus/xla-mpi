#include "pjrt_plugin/mpi_collectives.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpi.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla_mpi {

absl::StatusOr<MPI_Datatype> PrimitiveTypeToMpiType(xla::PrimitiveType element_type) {
    switch (element_type) {
        case xla::S8:
            return MPI_INT8_T;
        case xla::U8:
        case xla::PRED:
            return MPI_UINT8_T;
        case xla::S16:
            return MPI_INT16_T;
        case xla::U16:
            return MPI_UINT16_T;
        case xla::S32:
            return MPI_INT32_T;
        case xla::U32:
            return MPI_UINT32_T;
        case xla::S64:
            return MPI_INT64_T;
        case xla::U64:
            return MPI_UINT64_T;
        case xla::F32:
            return MPI_FLOAT;
        case xla::F64:
            return MPI_DOUBLE;
        case xla::C64:
            return MPI_C_COMPLEX;
        case xla::C128:
            return MPI_C_DOUBLE_COMPLEX;
        default:
            return absl::InvalidArgumentError(
                absl::StrCat("Unsupported primitive type for reduction: ",
                             xla::PrimitiveType_Name(element_type)));
    }
}

namespace {
bool MpiTypeIsComplex(MPI_Datatype type) {
    return type == MPI_C_COMPLEX || type == MPI_C_DOUBLE_COMPLEX;
}
}  // namespace

absl::StatusOr<MPI_Op> ReductionKindToMpiOp(xla::ReductionKind reduction_kind, MPI_Datatype type) {
    switch (reduction_kind) {
        case xla::ReductionKind::SUM:
            return MPI_SUM;
        case xla::ReductionKind::PRODUCT:
            return MPI_PROD;
        case xla::ReductionKind::MIN:
            if (!MpiTypeIsComplex(type)) return MPI_MIN;
            return absl::InvalidArgumentError("MIN reduction not supported for complex types");
        case xla::ReductionKind::MAX:
            if (!MpiTypeIsComplex(type)) return MPI_MAX;
            return absl::InvalidArgumentError("MAX reduction not supported for complex types");
        default:
            return absl::InvalidArgumentError("Unknown reduction kind");
    }
}

absl::Status MpiErrorToAbslStatus(int error) {
    if (error != MPI_SUCCESS) {
        char error_str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(error, error_str, &len);
        return absl::UnknownError(absl::StrCat("MPI error: ", std::string(error_str, len)));
    }
    return absl::OkStatus();
}

MpiCommunicator::MpiCommunicator(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &mpi_rank_);
    MPI_Comm_size(comm_, &mpi_size_);
}

MpiCommunicator::~MpiCommunicator() {
    if (comm_ != MPI_COMM_WORLD) MPI_Comm_free(&comm_);
}

xla::Future<> MpiCommunicator::AllReduce(::stream_executor::DeviceAddressBase send_buffer,
                                         ::stream_executor::DeviceAddressBase recv_buffer,
                                         xla::PrimitiveType dtype, size_t count,
                                         xla::ReductionKind reduction_kind,
                                         const Executor& executor) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    absl::StatusOr<MPI_Op> op = ReductionKindToMpiOp(reduction_kind, *type);
    if (!op.ok()) return op.status();
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Allreduce(send_buffer.opaque(), recv_buffer.opaque(), count, *type, *op, comm_));
    if (!status.ok()) return status;
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::CollectivePermute(::stream_executor::DeviceAddressBase send_buffer,
                                                  ::stream_executor::DeviceAddressBase recv_buffer,
                                                  xla::PrimitiveType dtype, size_t count,
                                                  std::optional<xla::RankId> source_rank,
                                                  absl::Span<const xla::RankId> target_ranks,
                                                  const Executor& executor) {
    // TODO: Check if it's OK to use 0 as a tag. Rationale is that there is
    // only one CollectivePermute at a time, but unsure if this is true
    int tag = 0;
    const int rank = mpi_rank_;
    size_t num_bytes = count * xla::primitive_util::ByteWidth(dtype);
    std::vector<MPI_Request> requests;

    if (source_rank) {
        if (source_rank->value() == rank) {
            std::memcpy(recv_buffer.opaque(), send_buffer.opaque(), num_bytes);
        } else {
            requests.emplace_back();
            absl::Status status = MpiErrorToAbslStatus(
                MPI_Irecv(recv_buffer.opaque(), num_bytes, MPI_BYTE, source_rank->value(), tag, comm_,
                         &requests.back()));
            if (!status.ok()) return status;
        }
    } else {
        std::memset(recv_buffer.opaque(), 0, num_bytes);
    }

    for (xla::RankId target : target_ranks) {
        if (target.value() != rank) {
            requests.emplace_back();
            absl::Status status = MpiErrorToAbslStatus(
                MPI_Isend(send_buffer.opaque(), num_bytes, MPI_BYTE, target.value(), tag, comm_,
                         &requests.back()));
            if (!status.ok()) return status;
        }
    }

    for (auto& request : requests) {
        absl::Status status = MpiErrorToAbslStatus(MPI_Wait(&request, MPI_STATUS_IGNORE));
        if (!status.ok()) return status;
    }
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::AllToAll(
    absl::InlinedVector<::stream_executor::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<::stream_executor::DeviceAddressBase, 4> recv_buffers,
    xla::PrimitiveType dtype, size_t count, const Executor& executor) {
    const int rank = mpi_rank_;
    const int size = mpi_size_;
    if (static_cast<int>(send_buffers.size()) != size || static_cast<int>(recv_buffers.size()) != size) {
        return absl::InvalidArgumentError("AllToAll: expected one buffer per rank");
    }

    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    size_t chunk_bytes = count * xla::primitive_util::ByteWidth(dtype);

    std::memcpy(recv_buffers[rank].opaque(), send_buffers[rank].opaque(), chunk_bytes);

    for (int i = 1; i < size; ++i) {
        int send_rank = (rank + i) % size;
        int recv_rank = (rank + size - i) % size;
        absl::Status status = MpiErrorToAbslStatus(MPI_Sendrecv(
            send_buffers[send_rank].opaque(), count, *type, send_rank, /*sendtag=*/0,
            recv_buffers[recv_rank].opaque(), count, *type, recv_rank, /*recvtag=*/0, comm_,
            MPI_STATUS_IGNORE));
        if (!status.ok()) return status;
    }
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::AllGather(::stream_executor::DeviceAddressBase send_buffer,
                                         ::stream_executor::DeviceAddressBase recv_buffer,
                                         xla::PrimitiveType dtype, size_t count,
                                         const Executor& executor) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Allgather(send_buffer.opaque(), count, *type, recv_buffer.opaque(), count, *type, comm_));
    if (!status.ok()) return status;
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::ReduceScatter(::stream_executor::DeviceAddressBase send_buffer,
                                             ::stream_executor::DeviceAddressBase recv_buffer,
                                             xla::PrimitiveType dtype, size_t count,
                                             xla::ReductionKind reduction_kind,
                                             const Executor& executor) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    absl::StatusOr<MPI_Op> op = ReductionKindToMpiOp(reduction_kind, *type);
    if (!op.ok()) return op.status();
    std::vector<int> recvcounts(mpi_size_, static_cast<int>(count));
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Reduce_scatter(send_buffer.opaque(), recv_buffer.opaque(), recvcounts.data(), *type, *op, comm_));
    if (!status.ok()) return status;
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::Broadcast(::stream_executor::DeviceAddressBase,
                                         ::stream_executor::DeviceAddressBase,
                                         xla::PrimitiveType, size_t, xla::RankId,
                                         const Executor&) {
    return xla::Unimplemented("Broadcast is not implemented");
}

xla::Future<> MpiCommunicator::Send(::stream_executor::DeviceAddressBase send_buffer,
                                    xla::PrimitiveType dtype, size_t count, xla::RankId peer,
                                    const Executor& executor) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    constexpr int kTag = 0;
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Send(send_buffer.opaque(), count, *type, static_cast<int>(peer.value()), kTag, comm_));
    if (!status.ok()) return status;
    return absl::OkStatus();
}

xla::Future<> MpiCommunicator::Recv(::stream_executor::DeviceAddressBase recv_buffer,
                                    xla::PrimitiveType dtype, size_t count, xla::RankId peer,
                                    const Executor& executor) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(dtype);
    if (!type.ok()) return type.status();
    constexpr int kTag = 0;
    absl::Status status = MpiErrorToAbslStatus(MPI_Recv(recv_buffer.opaque(), count, *type,
                                                        static_cast<int>(peer.value()), kTag, comm_,
                                                        MPI_STATUS_IGNORE));
    if (!status.ok()) return status;
    return absl::OkStatus();
}

std::string MpiCommunicator::ToString() const {
    return absl::StrCat("MpiCommunicator [rank: ", mpi_rank_, " num_ranks: ", mpi_size_, "]");
}

namespace {

// Sets PJRT_NPROC which is read by xla::DefaultThreadPoolSize()
// Priority: our `XLA_MPI_CORES_PER_RANK` environment variable, then Slurm's per-task allocation. If
// neither is set then PJRT_NPROC is left untouched and XLA's own autodetection decides the pool size.
void ConfigureThreadPoolSizeFromEnvironment() {
    const char* explicit_override = std::getenv("XLA_MPI_CORES_PER_RANK");
    const char* slurm_hint = std::getenv("SLURM_CPUS_PER_TASK");
    const char* cores = explicit_override ? explicit_override : slurm_hint;
    if (cores != nullptr) {
        setenv("PJRT_NPROC", cores, /*overwrite=*/1);
    }
}

}  // namespace

void Mpi::Init() {
    ConfigureThreadPoolSizeFromEnvironment();

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            std::cerr << "WARNING: MPI_THREAD_MULTIPLE requested but not supported by this MPI "
                        "implementation"
                     << std::endl;
        }
    }
}

void Mpi::Finalize() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

absl::StatusOr<std::vector<std::unique_ptr<xla::Communicator>>>
MpiCollectives::CreateCommunicators(const xla::CliqueKey& clique_key,
                                    const std::optional<xla::CliqueIds>&,
                                    absl::Span<const DeviceRank> ranks, const Config&) {
    // this project defaults to
    // MULTIPLE instead (see Mpi::Init() in this file), so the restriction is
    // now conditional rather than absolute.
    int provided = 0;
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        int flag = 0;
        MPI_Is_thread_main(&flag);
        if (!flag) {
            return absl::UnknownError(
                "MPI: Communicator requested from a thread that is not the one MPI "
                "was initialized from, and MPI_THREAD_MULTIPLE was not provided. "
                "Concurrent collective dispatch is unsafe on this MPI build.");
        }
    }

    if (clique_key.num_devices() == 0) {
        return absl::InvalidArgumentError("MpiCollectives::CreateCommunicators: empty clique");
    }
    std::vector<int> group_ranks;
    group_ranks.reserve(clique_key.devices().size());
    for (xla::GlobalDeviceId id : clique_key.devices()) {
        group_ranks.push_back(UnpackCpuProcessIndex(id.value()));
    }

    absl::StatusOr<MPI_Comm> comm = CreateSubComm(std::move(group_ranks), /*tag=*/0);
    if (!comm.ok()) return comm.status();

    std::vector<std::unique_ptr<xla::Communicator>> communicators;
    communicators.reserve(ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
        communicators.push_back(std::make_unique<MpiCommunicator>(*comm));
    }
    return communicators;
}

Mpi& GetMpiSingleton() {
    static Mpi* const mpi = new Mpi();
    return *mpi;
}

}  // namespace xla_mpi
