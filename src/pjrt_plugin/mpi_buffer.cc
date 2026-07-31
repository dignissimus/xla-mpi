#include "pjrt_plugin/mpi_buffer.h"
#include "pjrt_plugin/pjrt_types.h"

#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace xla_mpi {

namespace {
size_t ElementByteSize(PJRT_Buffer_Type dtype) {
    switch (dtype) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
            return 8;
        default:
            throw std::runtime_error("Unsupported PJRT_Buffer_Type: " + std::to_string(dtype));
    }
}
}  // namespace

MpiBuffer::MpiBuffer(PJRT_Buffer_Type dtype, std::vector<int64_t> shape)
    : dtype_(dtype), shape_(std::move(shape)), data_(byte_size()) {}

std::unique_ptr<MpiBuffer> MpiBuffer::CreateFromHost(
    const void* data, PJRT_Buffer_Type dtype, const std::vector<int64_t>& shape) {
    auto buffer = std::make_unique<MpiBuffer>(dtype, shape);
    if (data != nullptr && buffer->byte_size() > 0) {
        std::memcpy(buffer->data(), data, buffer->byte_size());
    }
    return buffer;
}

size_t MpiBuffer::num_elements() const {
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
}

size_t MpiBuffer::element_size() const {
    return ElementByteSize(dtype_);
}

size_t MpiBuffer::byte_size() const {
    return num_elements() * element_size();
}

}  // namespace xla_mpi

PJRT_Error* MPI_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    delete args->buffer;
    return nullptr;
}

PJRT_Error* MPI_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
    args->type = args->buffer->buffer->dtype();
    return nullptr;
}

PJRT_Error* MPI_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
    const std::vector<int64_t>& shape = args->buffer->buffer->shape();
    args->dims = shape.data();
    args->num_dims = shape.size();
    return nullptr;
}

PJRT_Error* MPI_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    return MakeError("MPI_Buffer_UnpaddedDimensions not yet implemented");
}

PJRT_Error* MPI_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
}

PJRT_Error* MPI_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    return MakeError("MPI_Buffer_GetMemoryLayout not yet implemented");
}

PJRT_Error* MPI_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    args->on_device_size_in_bytes = args->buffer->buffer->byte_size();
    return nullptr;
}

PJRT_Error* MPI_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    args->device = args->buffer->device;
    return nullptr;
}

PJRT_Error* MPI_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    args->memory = args->buffer->memory;
    return nullptr;
}

PJRT_Error* MPI_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
    args->buffer->deleted = true;
    args->buffer->buffer.reset();
    return nullptr;
}

PJRT_Error* MPI_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
    args->is_deleted = args->buffer->deleted;
    return nullptr;
}

PJRT_Error* MPI_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
    return MakeError("MPI_Buffer_CopyToDevice not yet implemented");
}

PJRT_Error* MPI_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
    return MakeError("MPI_Buffer_CopyToMemory not yet implemented");
}

PJRT_Error* MPI_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    xla_mpi::MpiBuffer* buffer = args->src->buffer.get();
    if (buffer == nullptr) {
        return MakeError("MPI_Buffer_ToHostBuffer: buffer has been deleted");
    }
    size_t needed = buffer->byte_size();
    if (args->dst == nullptr) {
        args->dst_size = needed;
        args->event = new PJRT_Event{.ready = true};
        return nullptr;
    }
    if (args->dst_size < needed) {
        return MakeError("MPI_Buffer_ToHostBuffer: dst_size too small");
    }
    std::memcpy(args->dst, buffer->data(), needed);
    args->event = new PJRT_Event{.ready = true};
    return nullptr;
}

PJRT_Error* MPI_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
    args->is_on_cpu = true;
    return nullptr;
}

PJRT_Error* MPI_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
    args->event = new PJRT_Event{.ready = true};
    return nullptr;
}

PJRT_Error* MPI_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
    args->buffer_pointer = reinterpret_cast<uintptr_t>(args->buffer->buffer->data());
    return nullptr;
}

PJRT_Error* MPI_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPI_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPI_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
    args->device_memory_ptr = args->buffer->buffer->data();
    return nullptr;
}

PJRT_Error* MPI_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args) {
    return MakeError("MPI_CopyToDeviceStream_Destroy not yet implemented");
}
PJRT_Error* MPI_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args) {
    return MakeError("MPI_CopyToDeviceStream_AddChunk not yet implemented");
}
PJRT_Error* MPI_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args) {
    return MakeError("MPI_CopyToDeviceStream_TotalBytes not yet implemented");
}
PJRT_Error* MPI_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args) {
    return MakeError("MPI_CopyToDeviceStream_GranuleSize not yet implemented");
}
PJRT_Error* MPI_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args) {
    return MakeError("MPI_CopyToDeviceStream_CurrentBytes not yet implemented");
}
