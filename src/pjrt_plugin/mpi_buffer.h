#ifndef XLA_MPI_MPI_BUFFER_H_
#define XLA_MPI_MPI_BUFFER_H_

#include <vector>
#include <cstdint>
#include <memory>
#include "xla/pjrt/c/pjrt_c_api.h"

namespace xla_mpi {

class MpiBuffer {
public:
    MpiBuffer(PJRT_Buffer_Type dtype, std::vector<int64_t> shape);
    ~MpiBuffer() = default;

    static std::shared_ptr<MpiBuffer> CreateFromHost(void* data, PJRT_Buffer_Type dtype, const std::vector<int64_t>& shape);

    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }
    
    PJRT_Buffer_Type dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    size_t num_elements() const;
    size_t element_size() const;
    size_t byte_size() const;

private:
    PJRT_Buffer_Type dtype_;
    std::vector<int64_t> shape_;
    std::vector<uint8_t> data_;
};

} // namespace xla_mpi
#endif
