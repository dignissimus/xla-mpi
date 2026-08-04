#include "pjrt_plugin/mpi_process_group.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace xla_mpi {

absl::StatusOr<std::vector<int>> ResolveProcessGroupRanks(
    ProcessGroupStrategy strategy, absl::Span<const int64_t> groups, int64_t num_groups,
    int64_t num_partitions, absl::Span<const int64_t> device_assignment, int64_t my_replica,
    int64_t my_partition) {
    if (num_groups <= 0 || groups.size() % static_cast<size_t>(num_groups) != 0) {
        return absl::InvalidArgumentError("ResolveProcessGroupRanks: malformed groups");
    }
    int64_t group_size = static_cast<int64_t>(groups.size()) / num_groups;

    auto device_at = [&](int64_t replica, int64_t partition) -> absl::StatusOr<int> {
        int64_t idx = replica * num_partitions + partition;
        if (idx < 0 || idx >= static_cast<int64_t>(device_assignment.size())) {
            return absl::InvalidArgumentError(
                absl::StrCat("ResolveProcessGroupRanks::DeviceAt: (replica=", replica,
                            ", partition=", partition, ") out of range"));
        }
        return UnpackCpuProcessIndex(device_assignment[idx]);
    };

    auto append_rank = [&](std::vector<int>& ranks, int64_t r, int64_t p) -> absl::Status {
        absl::StatusOr<int> d = device_at(r, p);
        if (!d.ok()) return d.status();
        ranks.push_back(*d);
        return absl::OkStatus();
    };

    for (int64_t g = 0; g < num_groups; ++g) {
        absl::Span<const int64_t> row = groups.subspan(g * group_size, group_size);
        std::vector<int> ranks;

        switch (strategy) {
            case ProcessGroupStrategy::kCrossReplica: {
                // One group per (replica_group row, partition), restricted
                // to my own partition.
                if (std::find(row.begin(), row.end(), my_replica) == row.end()) continue;
                for (int64_t r : row) {
                    if (absl::Status s = append_rank(ranks, r, my_partition); !s.ok()) return s;
                }
                return ranks;
            }
            case ProcessGroupStrategy::kCrossPartition: {
                // One group per (replica, partition_group row), restricted
                // to my own replica.
                if (std::find(row.begin(), row.end(), my_partition) == row.end()) continue;
                for (int64_t p : row) {
                    if (absl::Status s = append_rank(ranks, my_replica, p); !s.ok()) return s;
                }
                return ranks;
            }
            case ProcessGroupStrategy::kCrossReplicaAndPartition: {
                // One group per row, spanning every partition of every
                // replica in the row.
                if (std::find(row.begin(), row.end(), my_replica) == row.end()) continue;
                for (int64_t p = 0; p < num_partitions; ++p) {
                    for (int64_t r : row) {
                        if (absl::Status s = append_rank(ranks, r, p); !s.ok()) return s;
                    }
                }
                return ranks;
            }
            case ProcessGroupStrategy::kFlattenedIds: {
                // One group per row of flattened (replica * num_partitions +
                // partition) ids.
                int64_t my_flat = my_replica * num_partitions + my_partition;
                if (std::find(row.begin(), row.end(), my_flat) == row.end()) continue;
                for (int64_t flat : row) {
                    if (absl::Status s =
                            append_rank(ranks, flat / num_partitions, flat % num_partitions);
                        !s.ok()) {
                        return s;
                    }
                }
                return ranks;
            }
        }
    }
    return absl::InvalidArgumentError(
        "ResolveProcessGroupRanks: this process is not covered by any group");
}

absl::StatusOr<MPI_Comm> CreateSubComm(std::vector<int> ranks, int tag) {
    std::sort(ranks.begin(), ranks.end());

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group sub_group;
    MPI_Group_incl(world_group, static_cast<int>(ranks.size()), ranks.data(), &sub_group);
    MPI_Comm sub_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, tag, &sub_comm);
    MPI_Group_free(&world_group);
    MPI_Group_free(&sub_group);

    return sub_comm;
}

}  // namespace xla_mpi
