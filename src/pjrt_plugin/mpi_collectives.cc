#include "pjrt_plugin/mpi_collectives.h"

#include <cstddef>
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

MpiCommunicator::MpiCommunicator(int color, int key) {
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm_);
    MPI_Comm_rank(comm_, &mpi_rank_);
    MPI_Comm_size(comm_, &mpi_size_);
}

MpiCommunicator::~MpiCommunicator() { MPI_Comm_free(&comm_); }

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

xla::Future<> MpiCommunicator::CollectivePermute(::stream_executor::DeviceAddressBase,
                                                  ::stream_executor::DeviceAddressBase,
                                                  xla::PrimitiveType, size_t,
                                                  std::optional<xla::RankId>,
                                                  absl::Span<const xla::RankId>,
                                                  const Executor&) {
    return xla::Unimplemented("CollectivePermute is not implemented");
}

xla::Future<> MpiCommunicator::AllToAll(
    absl::InlinedVector<::stream_executor::DeviceAddressBase, 4>,
    absl::InlinedVector<::stream_executor::DeviceAddressBase, 4>,
    xla::PrimitiveType, size_t, const Executor&) {
    return xla::Unimplemented("AllToAll is not implemented");
}

xla::Future<> MpiCommunicator::AllGather(::stream_executor::DeviceAddressBase,
                                         ::stream_executor::DeviceAddressBase,
                                         xla::PrimitiveType, size_t, const Executor&) {
    return xla::Unimplemented("AllGather is not implemented");
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

xla::Future<> MpiCommunicator::Send(::stream_executor::DeviceAddressBase, xla::PrimitiveType,
                                    size_t, xla::RankId, const Executor&) {
    return xla::Unimplemented("Send is not implemented");
}

xla::Future<> MpiCommunicator::Recv(::stream_executor::DeviceAddressBase, xla::PrimitiveType,
                                    size_t, xla::RankId, const Executor&) {
    return xla::Unimplemented("Recv is not implemented");
}

std::string MpiCommunicator::ToString() const {
    return absl::StrCat("MpiCommunicator [rank: ", mpi_rank_, " num_ranks: ", mpi_size_, "]");
}

void Mpi::Init() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
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
    int flag = 0;
    MPI_Is_thread_main(&flag);
    if (!flag) {
        return absl::UnknownError(
            "MPI: Communicator requested from a thread that is not the one MPI "
            "was initialized from. Multiple threads/devices per process are not "
            "yet supported.");
    }

    std::vector<std::unique_ptr<xla::Communicator>> communicators;
    communicators.reserve(ranks.size());
    for (const DeviceRank& device_rank : ranks) {
        size_t rank = device_rank.rank.value();
        int color;
        int key = 0;
        if (clique_key.num_devices() > 0) {
            color = static_cast<int>(clique_key.devices().at(0).value());
            key = static_cast<int>(rank);
        } else {
            color = MPI_UNDEFINED;
        }
        communicators.push_back(std::make_unique<MpiCommunicator>(color, key));
    }
    return communicators;
}

Mpi& GetMpiSingleton() {
    static Mpi* const mpi = new Mpi();
    return *mpi;
}

}  // namespace xla_mpi
