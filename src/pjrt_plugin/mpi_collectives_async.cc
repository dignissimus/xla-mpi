#include "pjrt_plugin/mpi_collectives_async.h"
#include "pjrt_plugin/mpi_collectives.h"
#include "pjrt_plugin/mpi_process_group.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
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

absl::StatusOr<int> ResolvePermuteRank(ProcessGroupStrategy strategy, int64_t raw_id,
                                       int64_t num_partitions,
                                       absl::Span<const int64_t> device_assignment,
                                       int64_t my_replica, int64_t my_partition) {
    int64_t replica = strategy == ProcessGroupStrategy::kCrossPartition ? my_replica : raw_id;
    int64_t partition = strategy == ProcessGroupStrategy::kCrossPartition ? raw_id : my_partition;
    int64_t idx = replica * num_partitions + partition;
    if (idx < 0 || idx >= static_cast<int64_t>(device_assignment.size())) {
        return absl::InvalidArgumentError("ResolvePermuteRank: resolved coordinate out of range");
    }
    return UnpackCpuProcessIndex(device_assignment[idx]);
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

absl::Status ReduceScatterStart(ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> recv,
                                int32_t reduction_kind, int64_t scatter_dimension, int32_t group_tag,
                                absl::Span<const int64_t> groups, int64_t num_groups,
                                int32_t process_group_strategy, int64_t num_partitions,
                                absl::Span<const int64_t> device_assignment, int64_t my_rank,
                                int64_t my_replica, int64_t my_partition, int64_t request_buffer) {
    (void)recv;
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(input.element_type());
    if (!type.ok()) return type.status();
    absl::StatusOr<xla::ReductionKind> kind = DecodeReductionKind(reduction_kind);
    if (!kind.ok()) return kind.status();
    absl::StatusOr<MPI_Op> op = ReductionKindToMpiOp(*kind, *type);
    if (!op.ok()) return op.status();

    absl::StatusOr<std::vector<int>> group_ranks = ResolveProcessGroupRanks(
        static_cast<ProcessGroupStrategy>(process_group_strategy), groups, num_groups,
        num_partitions, device_assignment, my_replica, my_partition);
    if (!group_ranks.ok()) return group_ranks.status();
    absl::StatusOr<MPI_Comm> comm = CreateSubComm(*group_ranks, group_tag);
    if (!comm.ok()) return comm.status();

    auto it = std::find(group_ranks->begin(), group_ranks->end(), my_rank);
    int64_t chunk_index = std::distance(group_ranks->begin(), it);

    void* scratch = ::operator new(input.size_bytes());
    std::vector<int64_t> dims(input.dimensions().begin(), input.dimensions().end());

    MPI_Request request;
    absl::Status status = MpiErrorToAbslStatus(MPI_Iallreduce(
        input.untyped_data(), scratch, input.element_count(), *type, *op, *comm, &request));
    if (!status.ok()) {
        ::operator delete(scratch);
        if (*comm != MPI_COMM_WORLD) MPI_Comm_free(&*comm);
        return status;
    }
    reinterpret_cast<MpiReduceScatterRequestBuffer*>(request_buffer)
        ->Store(request, scratch, std::move(dims), scatter_dimension, chunk_index, *type, *comm);
    return absl::OkStatus();
}

absl::Status ReduceScatterDone(ffi::AnyBuffer recv_buffer, ffi::Result<ffi::AnyBuffer> result,
                               int64_t request_buffer) {
    return reinterpret_cast<MpiReduceScatterRequestBuffer*>(request_buffer)->MpiWait(recv_buffer, result);
}

struct ResizedBlockTypes {
    MPI_Datatype block_type;
    MPI_Datatype resized_type;
};

absl::StatusOr<ResizedBlockTypes> BuildResizedBlockType(
    absl::Span<const int64_t> container_sizes, absl::Span<const int64_t> unit_sizes, int64_t dim,
    MPI_Datatype elem_type, size_t elem_byte_size) {
    int ndims = static_cast<int>(container_sizes.size());
    if (static_cast<int64_t>(unit_sizes.size()) != ndims || dim < 0 || dim >= ndims) {
        return absl::InvalidArgumentError("BuildResizedBlockType: shape/dim mismatch");
    }
    std::vector<int> container(container_sizes.begin(), container_sizes.end());
    std::vector<int> unit(unit_sizes.begin(), unit_sizes.end());
    std::vector<int> starts(ndims, 0);

    MPI_Datatype block_type;
    MPI_Type_create_subarray(ndims, container.data(), unit.data(), starts.data(), MPI_ORDER_C, elem_type,
                             &block_type);
    MPI_Type_commit(&block_type);

    int64_t stride_elems = 1;
    for (int64_t d = dim + 1; d < ndims; ++d) stride_elems *= container_sizes[d];
    auto extent = static_cast<MPI_Aint>(unit_sizes[dim] * stride_elems * elem_byte_size);

    MPI_Datatype resized_type;
    MPI_Type_create_resized(block_type, /*lb=*/0, extent, &resized_type);
    MPI_Type_commit(&resized_type);

    return ResizedBlockTypes{block_type, resized_type};
}

absl::Status AllGatherStart(ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> recv,
                            int64_t all_gather_dim, int32_t group_tag, absl::Span<const int64_t> groups,
                            int64_t num_groups, int32_t process_group_strategy, int64_t num_partitions,
                            absl::Span<const int64_t> device_assignment, int64_t my_replica,
                            int64_t my_partition, int64_t request_buffer) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(input.element_type());
    if (!type.ok()) return type.status();
    absl::StatusOr<MPI_Comm> comm =
        ResolveGroupComm(groups, num_groups, process_group_strategy, num_partitions,
                         device_assignment, my_replica, my_partition, group_tag);
    if (!comm.ok()) return comm.status();

    size_t elem_size = input.size_bytes() / input.element_count();
    absl::StatusOr<ResizedBlockTypes> block_types = BuildResizedBlockType(
        recv->dimensions(), input.dimensions(), all_gather_dim, *type, elem_size);
    if (!block_types.ok()) return block_types.status();

    std::vector<MPI_Datatype> types = {block_types->block_type, block_types->resized_type};
    std::vector<MPI_Request> requests(1);
    absl::Status status = MpiErrorToAbslStatus(
        MPI_Iallgather(input.untyped_data(), input.element_count(), *type, recv->untyped_data(), 1,
                      block_types->resized_type, *comm, &requests[0]));
    if (!status.ok()) {
        for (MPI_Datatype t : types) MPI_Type_free(&t);
        if (*comm != MPI_COMM_WORLD) MPI_Comm_free(&*comm);
        return status;
    }
    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests), std::move(types), *comm);
    return absl::OkStatus();
}

absl::Status AllGatherDone(ffi::AnyBuffer recv_buffer, ffi::Result<ffi::AnyBuffer> result,
                          int64_t request_buffer) {
    (void)recv_buffer;
    return reinterpret_cast<MpiRequestBuffer*>(request_buffer)->MpiWait();
}

absl::Status CollectivePermuteStart(ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> recv,
                                    absl::Span<const int64_t> groups, int32_t process_group_strategy,
                                    int64_t num_partitions, absl::Span<const int64_t> device_assignment,
                                    int64_t my_rank, int64_t my_replica, int64_t my_partition,
                                    int64_t request_buffer) {
    // TODO: Check if it's OK to use 0 as a tag. Rationale is that there is
    // only one CollectivePermute at a time, but unsure if this is true
    constexpr int kTag = 0;
    size_t num_bytes = input.size_bytes();
    auto strategy = static_cast<ProcessGroupStrategy>(process_group_strategy);

    std::optional<int> source_rank;
    std::vector<int> target_ranks;
    for (size_t i = 0; i + 1 < groups.size(); i += 2) {
        int64_t source_id = groups[i];
        int64_t target_id = groups[i + 1];
        absl::StatusOr<int> source_dev = ResolvePermuteRank(strategy, source_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        absl::StatusOr<int> target_dev = ResolvePermuteRank(strategy, target_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        if (!source_dev.ok()) return source_dev.status();
        if (!target_dev.ok()) return target_dev.status();
        // ResolvePermuteRank resolves a raw id relative to *my own* other
        // coordinate -- so *source_dev only actually names "the source of
        // this pair" if I'm the pair's target (source_id's role), and vice
        // versa; check against my_rank the same way the original
        // rank-equals-replica-id logic did, just through the resolved rank.
        if (*target_dev == my_rank) source_rank = *source_dev;
        if (*source_dev == my_rank) target_ranks.push_back(*target_dev);
    }

    std::vector<MPI_Request> requests;

    if (source_rank) {
        if (*source_rank == my_rank) {
            std::memcpy(recv->untyped_data(), input.untyped_data(), num_bytes);
        } else {
            requests.emplace_back();
            absl::Status status = MpiErrorToAbslStatus(
                MPI_Irecv(recv->untyped_data(), num_bytes, MPI_BYTE, *source_rank, kTag,
                         MPI_COMM_WORLD, &requests.back()));
            if (!status.ok()) return status;
        }
    } else {
        std::memset(recv->untyped_data(), 0, num_bytes);
    }

    for (int target : target_ranks) {
        if (target != my_rank) {
            requests.emplace_back();
            absl::Status status = MpiErrorToAbslStatus(
                MPI_Isend(input.untyped_data(), num_bytes, MPI_BYTE, target, kTag, MPI_COMM_WORLD,
                         &requests.back()));
            if (!status.ok()) return status;
        }
    }

    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests));
    return absl::OkStatus();
}

absl::Status CollectivePermuteDone(ffi::AnyBuffer recv_buffer, ffi::Result<ffi::AnyBuffer> result,
                                   int64_t request_buffer) {
    (void)recv_buffer;
    return reinterpret_cast<MpiRequestBuffer*>(request_buffer)->MpiWait();
}

absl::Status AllToAllStart(ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> recv,
                           int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
                           absl::Span<const int64_t> groups, int64_t num_groups,
                           int32_t process_group_strategy, int64_t num_partitions,
                           absl::Span<const int64_t> device_assignment, int64_t my_replica,
                           int64_t my_partition, int64_t request_buffer) {
    absl::StatusOr<MPI_Datatype> type = PrimitiveTypeToMpiType(input.element_type());
    if (!type.ok()) return type.status();

    absl::StatusOr<std::vector<int>> group_ranks = ResolveProcessGroupRanks(
        static_cast<ProcessGroupStrategy>(process_group_strategy), groups, num_groups,
        num_partitions, device_assignment, my_replica, my_partition);
    if (!group_ranks.ok()) return group_ranks.status();
    if (static_cast<int64_t>(group_ranks->size()) != split_count) {
        return absl::InvalidArgumentError(
            absl::StrCat("AllToAll: split_count (", split_count, ") must equal group size (",
                         group_ranks->size(), ")"));
    }

    absl::Span<const int64_t> in_dims = input.dimensions();
    absl::Span<const int64_t> out_dims = recv->dimensions();
    int ndims = static_cast<int>(in_dims.size());
    std::vector<int> in_sizes(in_dims.begin(), in_dims.end());
    std::vector<int> out_sizes(out_dims.begin(), out_dims.end());

    int64_t send_chunk = in_dims[split_dimension] / split_count;
    int64_t recv_chunk = out_dims[concat_dimension] / split_count;

    std::vector<MPI_Request> requests;
    std::vector<MPI_Datatype> types;
    constexpr int kTag = 0;

    auto fail = [&](absl::Status status) {
        for (MPI_Datatype t : types) MPI_Type_free(&t);
        return status;
    };

    for (int64_t i = 0; i < split_count; ++i) {
        int peer = (*group_ranks)[i];

        std::vector<int> send_subsizes = in_sizes;
        send_subsizes[split_dimension] = static_cast<int>(send_chunk);
        std::vector<int> send_starts(ndims, 0);
        send_starts[split_dimension] = static_cast<int>(i * send_chunk);

        std::vector<int> recv_subsizes = out_sizes;
        recv_subsizes[concat_dimension] = static_cast<int>(recv_chunk);
        std::vector<int> recv_starts(ndims, 0);
        recv_starts[concat_dimension] = static_cast<int>(i * recv_chunk);

        MPI_Datatype send_type;
        MPI_Type_create_subarray(ndims, in_sizes.data(), send_subsizes.data(), send_starts.data(),
                                 MPI_ORDER_C, *type, &send_type);
        MPI_Type_commit(&send_type);
        types.push_back(send_type);

        MPI_Datatype recv_type;
        MPI_Type_create_subarray(ndims, out_sizes.data(), recv_subsizes.data(), recv_starts.data(),
                                 MPI_ORDER_C, *type, &recv_type);
        MPI_Type_commit(&recv_type);
        types.push_back(recv_type);

        requests.emplace_back();
        absl::Status send_status = MpiErrorToAbslStatus(MPI_Isend(
            input.untyped_data(), 1, send_type, peer, kTag, MPI_COMM_WORLD, &requests.back()));
        if (!send_status.ok()) return fail(send_status);

        requests.emplace_back();
        absl::Status recv_status = MpiErrorToAbslStatus(MPI_Irecv(
            recv->untyped_data(), 1, recv_type, peer, kTag, MPI_COMM_WORLD, &requests.back()));
        if (!recv_status.ok()) return fail(recv_status);
    }

    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests), std::move(types));
    return absl::OkStatus();
}

absl::Status AllToAllDone(ffi::AnyBuffer recv_buffer, ffi::Result<ffi::AnyBuffer> result,
                         int64_t request_buffer) {
    (void)recv_buffer;
    return reinterpret_cast<MpiRequestBuffer*>(request_buffer)->MpiWait();
}

absl::Status SendStart(ffi::AnyBuffer input, ffi::Token incoming_token,
                       ffi::Result<ffi::Token> outgoing_token, int64_t channel_id,
                       absl::Span<const int64_t> groups, int32_t process_group_strategy,
                       int64_t num_partitions, absl::Span<const int64_t> device_assignment,
                       int64_t my_rank, int64_t my_replica, int64_t my_partition,
                       int64_t request_buffer) {
    (void)incoming_token;
    (void)outgoing_token;
    int tag = static_cast<int>(channel_id);
    auto strategy = static_cast<ProcessGroupStrategy>(process_group_strategy);

    std::vector<MPI_Request> requests;
    for (size_t i = 0; i + 1 < groups.size(); i += 2) {
        int64_t source_id = groups[i];
        int64_t target_id = groups[i + 1];
        absl::StatusOr<int> source_dev = ResolvePermuteRank(strategy, source_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        absl::StatusOr<int> target_dev = ResolvePermuteRank(strategy, target_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        if (!source_dev.ok()) return source_dev.status();
        if (!target_dev.ok()) return target_dev.status();
        if (*source_dev != my_rank) continue;

        requests.emplace_back();
        absl::Status status = MpiErrorToAbslStatus(
            MPI_Isend(input.untyped_data(), input.size_bytes(), MPI_BYTE, *target_dev, tag,
                     MPI_COMM_WORLD, &requests.back()));
        if (!status.ok()) return status;
    }

    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests));
    return absl::OkStatus();
}

absl::Status SendDone(ffi::Token incoming_token, ffi::Result<ffi::Token> outgoing_token,
                      int64_t request_buffer) {
    (void)incoming_token;
    (void)outgoing_token;
    return reinterpret_cast<MpiRequestBuffer*>(request_buffer)->MpiWait();
}

absl::Status RecvStart(ffi::Token incoming_token, ffi::Result<ffi::AnyBuffer> recv,
                       ffi::Result<ffi::Token> outgoing_token, int64_t channel_id,
                       absl::Span<const int64_t> groups, int32_t process_group_strategy,
                       int64_t num_partitions, absl::Span<const int64_t> device_assignment,
                       int64_t my_rank, int64_t my_replica, int64_t my_partition,
                       int64_t request_buffer) {
    (void)incoming_token;
    (void)outgoing_token;
    int tag = static_cast<int>(channel_id);
    auto strategy = static_cast<ProcessGroupStrategy>(process_group_strategy);

    std::vector<MPI_Request> requests;
    bool found = false;
    for (size_t i = 0; i + 1 < groups.size(); i += 2) {
        int64_t source_id = groups[i];
        int64_t target_id = groups[i + 1];
        absl::StatusOr<int> source_dev = ResolvePermuteRank(strategy, source_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        absl::StatusOr<int> target_dev = ResolvePermuteRank(strategy, target_id, num_partitions,
                                                            device_assignment, my_replica, my_partition);
        if (!source_dev.ok()) return source_dev.status();
        if (!target_dev.ok()) return target_dev.status();
        if (*target_dev != my_rank) continue;

        found = true;
        requests.emplace_back();
        absl::Status status = MpiErrorToAbslStatus(
            MPI_Irecv(recv->untyped_data(), recv->size_bytes(), MPI_BYTE, *source_dev, tag,
                     MPI_COMM_WORLD, &requests.back()));
        if (!status.ok()) return status;
    }
    if (!found) {
        std::memset(recv->untyped_data(), 0, recv->size_bytes());
    }

    reinterpret_cast<MpiRequestBuffer*>(request_buffer)->Store(std::move(requests));
    return absl::OkStatus();
}

absl::Status RecvDone(ffi::AnyBuffer recv_buffer, ffi::Token incoming_token,
                      ffi::Result<ffi::AnyBuffer> result, ffi::Result<ffi::Token> outgoing_token,
                      int64_t request_buffer) {
    (void)recv_buffer;
    (void)incoming_token;
    (void)outgoing_token;
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

XLA_FFI_DEFINE_HANDLER(kReduceScatterStart, ReduceScatterStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int32_t>("reduction_kind")
                           .Attr<int64_t>("scatter_dimension")
                           .Attr<int32_t>("group_tag")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int64_t>("num_groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_rank")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kReduceScatterDone, ReduceScatterDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kAllGatherStart, AllGatherStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("all_gather_dim")
                           .Attr<int32_t>("group_tag")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int64_t>("num_groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kAllGatherDone, AllGatherDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kCollectivePermuteStart, CollectivePermuteStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_rank")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kCollectivePermuteDone, CollectivePermuteDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kAllToAllStart, AllToAllStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("split_dimension")
                           .Attr<int64_t>("concat_dimension")
                           .Attr<int64_t>("split_count")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int64_t>("num_groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kAllToAllDone, AllToAllDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allreduce_start", "Host", kAllReduceStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allreduce_done", "Host", kAllReduceDone);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.reduce_scatter_start", "Host",
                         kReduceScatterStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.reduce_scatter_done", "Host",
                         kReduceScatterDone);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allgather_start", "Host", kAllGatherStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.allgather_done", "Host", kAllGatherDone);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.collective_permute_start", "Host",
                         kCollectivePermuteStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.collective_permute_done", "Host",
                         kCollectivePermuteDone);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.all_to_all_start", "Host", kAllToAllStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.all_to_all_done", "Host", kAllToAllDone);

XLA_FFI_DEFINE_HANDLER(kSendStart, SendStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::Token>()
                           .Ret<ffi::Token>()
                           .Attr<int64_t>("channel_id")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_rank")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kSendDone, SendDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Token>()
                           .Ret<ffi::Token>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kRecvStart, RecvStart,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Token>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::Token>()
                           .Attr<int64_t>("channel_id")
                           .Attr<absl::Span<const int64_t>>("groups")
                           .Attr<int32_t>("process_group_strategy")
                           .Attr<int64_t>("num_partitions")
                           .Attr<absl::Span<const int64_t>>("device_assignment")
                           .Attr<int64_t>("my_rank")
                           .Attr<int64_t>("my_replica")
                           .Attr<int64_t>("my_partition")
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_DEFINE_HANDLER(kRecvDone, RecvDone,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::Token>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::Token>()
                           .Attr<int64_t>("request_buffer"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.send_start", "Host", kSendStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.send_done", "Host", kSendDone);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.recv_start", "Host", kRecvStart);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_mpi.recv_done", "Host", kRecvDone);

}  // namespace

}  // namespace xla_mpi
