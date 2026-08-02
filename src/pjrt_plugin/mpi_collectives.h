#ifndef XLA_MPI_MPI_COLLECTIVES_H_
#define XLA_MPI_MPI_COLLECTIVES_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mpi.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/xla_data.pb.h"

namespace xla_mpi {

class MpiCommunicator : public xla::Communicator {
public:
    MpiCommunicator(int color, int key);
    ~MpiCommunicator() override;

    xla::Future<> AllReduce(::stream_executor::DeviceAddressBase send_buffer,
                            ::stream_executor::DeviceAddressBase recv_buffer,
                            xla::PrimitiveType dtype, size_t count,
                            xla::ReductionKind reduction_kind,
                            const Executor& executor) override;

    xla::Future<> CollectivePermute(::stream_executor::DeviceAddressBase send_buffer,
                                    ::stream_executor::DeviceAddressBase recv_buffer,
                                    xla::PrimitiveType dtype, size_t count,
                                    std::optional<xla::RankId> source_rank,
                                    absl::Span<const xla::RankId> target_ranks,
                                    const Executor& executor) override;

    xla::Future<> AllToAll(
        absl::InlinedVector<::stream_executor::DeviceAddressBase, 4> send_buffers,
        absl::InlinedVector<::stream_executor::DeviceAddressBase, 4> recv_buffers,
        xla::PrimitiveType dtype, size_t count, const Executor& executor) override;

    xla::Future<> AllGather(::stream_executor::DeviceAddressBase send_buffer,
                            ::stream_executor::DeviceAddressBase recv_buffer,
                            xla::PrimitiveType dtype, size_t count,
                            const Executor& executor) override;

    xla::Future<> ReduceScatter(::stream_executor::DeviceAddressBase send_buffer,
                                ::stream_executor::DeviceAddressBase recv_buffer,
                                xla::PrimitiveType dtype, size_t count,
                                xla::ReductionKind reduction_kind,
                                const Executor& executor) override;

    xla::Future<> Broadcast(::stream_executor::DeviceAddressBase send_buffer,
                            ::stream_executor::DeviceAddressBase recv_buffer,
                            xla::PrimitiveType dtype, size_t count,
                            xla::RankId root, const Executor& executor) override;

    xla::Future<> Send(::stream_executor::DeviceAddressBase send_buffer,
                       xla::PrimitiveType dtype, size_t count, xla::RankId peer,
                       const Executor& executor) override;

    xla::Future<> Recv(::stream_executor::DeviceAddressBase recv_buffer,
                       xla::PrimitiveType dtype, size_t count, xla::RankId peer,
                       const Executor& executor) override;

    absl::StatusOr<size_t> NumRanks() const override { return mpi_size_; }

    std::string ToString() const override;

private:
    MPI_Comm comm_;
    int mpi_rank_;
    int mpi_size_;
};

class MpiCollectives : public xla::cpu::CpuCollectives {
public:
    absl::StatusOr<std::vector<std::unique_ptr<xla::Communicator>>>
    CreateCommunicators(const xla::CliqueKey& clique_key,
                        const std::optional<xla::CliqueIds>& clique_ids,
                        absl::Span<const DeviceRank> ranks,
                        const Config& config) final;
};

// Owns this plugin's MPI process lifecycle (MPI_Init_thread/MPI_Finalize,
// guarded so both are safe to call more than once) and the MpiCollectives
// instance handed to XLA wherever a xla::cpu::CpuCollectives is needed.
// Composition, not inheritance -- Mpi does more than the CpuCollectives
// interface asks for (lifecycle isn't part of that contract), so it owns
// one instead of being one.
class Mpi {
public:
    void Init();
    void Finalize();

    MpiCollectives& collectives() { return collectives_; }

private:
    MpiCollectives collectives_;
};

// Process-global singleton, initialized once as early as possible (see
// GetPjrtApi()/MPI_Plugin_Initialize) and finalized once at matching
// teardown (see MPI_Client_Destroy).
Mpi& GetMpiSingleton();

}  // namespace xla_mpi

#endif  // XLA_MPI_MPI_COLLECTIVES_H_
