#ifndef XLA_MPI_MPI_PROCESS_GROUP_H_
#define XLA_MPI_MPI_PROCESS_GROUP_H_

#include <cstdint>
#include <vector>

#include <mpi.h>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xla_mpi {

// StableHLO process groups
enum class ProcessGroupStrategy {
    kCrossReplica,
    kCrossPartition,
    kCrossReplicaAndPartition,
    kFlattenedIds,
};

absl::StatusOr<std::vector<int>> ResolveProcessGroupRanks(
    ProcessGroupStrategy strategy, absl::Span<const int64_t> groups, int64_t num_groups,
    int64_t num_partitions, absl::Span<const int64_t> device_assignment, int64_t my_replica,
    int64_t my_partition);

absl::StatusOr<MPI_Comm> CreateSubComm(std::vector<int> ranks, int tag);

}  // namespace xla_mpi

#endif  // XLA_MPI_MPI_PROCESS_GROUP_H_
