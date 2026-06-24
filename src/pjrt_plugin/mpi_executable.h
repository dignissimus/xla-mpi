#ifndef XLA_MPI_MPI_EXECUTABLE_H_
#define XLA_MPI_MPI_EXECUTABLE_H_

#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/stablehlo_parser.h"
#include "pjrt_plugin/mpi_buffer.h"

namespace xla_mpi {

struct OutputInfo {
    int dtype;
    std::vector<int64_t> shape;
};

struct MpiExecuteResult {
    std::vector<MpiBuffer*> buffers;
    std::string error_message;
};

class MpiExecutable {
public:
    static std::unique_ptr<MpiExecutable> Create(ParsedModule parsed_module);
    ~MpiExecutable();

    bool IsValid() const;
    std::string error() const;
    size_t num_outputs() const;

    MpiExecuteResult Execute(const std::vector<MpiBuffer*>& inputs);

private:
    MpiExecutable() = default;

    ParsedModule parsed_module_;
    bool valid_ = false;
    std::string error_;
    size_t num_outputs_ = 0;
    std::vector<OutputInfo> output_info_;
};

}  // namespace xla_mpi

#endif  // XLA_MPI_MPI_EXECUTABLE_H_
