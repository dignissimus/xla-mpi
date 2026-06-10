#pragma once
#include <mpi.h>

namespace xla_mpi {

class MpiDevice {
private:
    int id_;

public:
    explicit MpiDevice(int id) : id_(id) {}

    int id() const { return id_; }
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
