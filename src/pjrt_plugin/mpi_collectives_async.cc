#include "pjrt_plugin/mpi_collectives_async.h"
#include "pjrt_plugin/mpi_collectives.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <mpi.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace xla_mpi {

namespace {

namespace ffi = ::xla::ffi;

absl::StatusOr<xla::ReductionKind> DecodeReductionKind(int32_t reduction_kind) {
    switch (reduction_kind) {
        case static_cast<int32_t>(xla::ReductionKind::SUM):
        case static_cast<int32_t>(xla::ReductionKind::PRODUCT):
        case static_cast<int32_t>(xla::ReductionKind::MIN):
        case static_cast<int32_t>(xla::ReductionKind::MAX):
            return static_cast<xla::ReductionKind>(reduction_kind);
        default:
            return absl::InvalidArgumentError(
                absl::StrCat("Unknown reduction_kind: ", reduction_kind));
    }
}

absl::StatusOr<MPI_Comm> ResolveGroupComm(absl::Span<const int64_t> groups, int64_t num_groups,
                                          int32_t process_group_strategy, int64_t num_partitions,
                                          absl::Span<const int64_t> device_assignment,
                                          int64_t my_replica, int64_t my_partition,
                                          int32_t group_tag) {
    absl::StatusOr<std::vector<int>> ranks = ResolveProcessGroupRanks(
        static_cast<ProcessGroupStrategy>(process_group_strategy), groups, num_groups,
        num_partitions, device_assignment, my_replica, my_partition);
    if (!ranks.ok()) return ranks.status();
    return CreateSubComm(std::move(*ranks), group_tag);
}

absl::Status AllReduceStart(ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> recv,
                            int32_t reduction_kind, int32_t group_tag,
                            absl::Span<const int64_t> groups, int64_t num_groups,
                            int32_t process_group_strategy, int64_t num_partitions,
                            absl::Span<const int64_t> device_assignment, int64_t my_replica,
                            int64_t my_partition, int64_t request_buffer) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(input.element_type());
    if (!type.ok()) return type.status();
    absl::StatusOr<xla::ReductionKind> kind = DecodeReductionKind(reduction_kind);
    if (!kind.ok()) return kind.status();
    absl::StatusOr<MPI_Op> op = ReductionKindToMpiOp(*kind, *type);
    if (!op.ok()) return op.status();
    absl::StatusOr<MPI_Comm> comm =
        ResolveGroupComm(groups, num_groups, process_group_strategy, num_partitions,
                         device_assignment, my_replica, my_partition, group_tag);
    if (!comm.ok()) return comm.status();

    std::vector<MPI_Request> requests(1);
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Iallreduce(input.untyped_data(), recv->untyped_data(), input.element_count(), *type,
                       *op, *comm, &requests[0]));
    if (!status.ok()) {
        if (*comm != MPI_COMM_WORLD) MPI_Comm_free(&*comm);
        return status;
    }
    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests), {}, *comm);
    return absl::OkStatus();
}

absl::Status AllReduceDone(ffi::AnyBuffer recv_buffer, ffi::Result<ffi::AnyBuffer> result,
                          int64_t request_buffer) {
    (void)recv_buffer;
    return reinterpret_cast<MpiRequestBuffer*>(request_buffer)->MpiWait();
}

XLA_FFI_DEFINE_HANDLER(kAllReduceStart, AllReduceStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int32_t>("reduction_kind")
                           .Attr<int32_t>("group_tag")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int64_t>("num_groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kAllReduceDone, AllReduceDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allreduce_start", "Host", kAllReduceStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allreduce_done", "Host", kAllReduceDone);

}  // namespace

}  // namespace xla_mpi
