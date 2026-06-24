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

#include "pjrt_plugin/pjrt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

PJRT_Error* MPI_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);
PJRT_Error* MPI_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args);
PJRT_Error* MPI_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args);
PJRT_Error* MPI_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args);
PJRT_Error* MPI_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args);
PJRT_Error* MPI_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args);
PJRT_Error* MPI_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* MPI_Buffer_Device(PJRT_Buffer_Device_Args* args);
PJRT_Error* MPI_Buffer_Memory(PJRT_Buffer_Memory_Args* args);
PJRT_Error* MPI_Buffer_Delete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* MPI_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* MPI_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args);
PJRT_Error* MPI_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args);
PJRT_Error* MPI_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args);
PJRT_Error* MPI_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);
PJRT_Error* MPI_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);
PJRT_Error* MPI_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args);
PJRT_Error* MPI_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);
PJRT_Error* MPI_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);
PJRT_Error* MPI_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);

PJRT_Error* MPI_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args);
PJRT_Error* MPI_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args);
PJRT_Error* MPI_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args);
PJRT_Error* MPI_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args);
PJRT_Error* MPI_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args);

#ifdef __cplusplus
}
#endif

#endif
