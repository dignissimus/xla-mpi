#ifndef XLA_MPI_MPI_PROCESS_GROUP_H_
#define XLA_MPI_MPI_PROCESS_GROUP_H_

#include <cstdint>
#include <vector>

#include <mpi.h>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xla_mpi {

constexpr int64_t kMaxCpuDevicesPerProcess = 1 << 11;  // 2048

inline int64_t PackCpuDeviceId(int process_id, int local_device_id = 0) {
    return static_cast<int64_t>(process_id) * kMaxCpuDevicesPerProcess + local_device_id;
}

inline int UnpackCpuProcessIndex(int64_t global_device_id) {
    return static_cast<int>(global_device_id / kMaxCpuDevicesPerProcess);
}

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
