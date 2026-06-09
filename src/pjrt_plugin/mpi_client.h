#pragma once
#include <mpi.h>

namespace xla_mpi {

class MpiDevice {
public:
    int id() const { return 0; }
};

class MpiClient {
private:
    int rank_;
    int world_size_;
public:
    MpiClient() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
};

class MpiBuffer {};
class MpiExecutable {};

} // namespace xla_mpi
