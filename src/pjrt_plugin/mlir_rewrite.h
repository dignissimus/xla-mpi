#ifndef XLA_MPI_MLIR_REWRITE_H_
#define XLA_MPI_MLIR_REWRITE_H_

#include <cstdint>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace xla_mpi {

struct ProgramInfo {
    int64_t num_replicas = 1;
    int64_t num_partitions = 1;
    std::vector<int64_t> device_assignment_flat;  // [replica * num_partitions + partition]

    int64_t my_rank = 0;
    int64_t my_replica = 0;
    int64_t my_partition = 0;
};

// Walks `entry` replacing every recognized collective/p2p op (AllReduce,
// ReduceScatter, AllGather, CollectivePermute, AllToAll, Send, Recv) with an
// async Start/Done custom-call pair (see mpi_collectives_async.cc for the
// FFI handlers). Anything that doesn't match its op's recognizer is left
// untouched -- still correct (synchronous), just without the overlap
// opportunity. Must run before the module is handed to XLA's HLO
// conversion/compilation.
void RewriteCollectivesAsAsync(mlir::func::FuncOp entry, const ProgramInfo& program_info);

}  // namespace xla_mpi

#endif  // XLA_MPI_MLIR_REWRITE_H_
