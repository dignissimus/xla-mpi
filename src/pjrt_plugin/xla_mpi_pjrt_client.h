#ifndef XLA_MPI_XLA_MPI_PJRT_CLIENT_H_
#define XLA_MPI_XLA_MPI_PJRT_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla_mpi {

class XlaMpiPjRtClient : public xla::PjRtClient {
public:
    explicit XlaMpiPjRtClient(std::unique_ptr<xla::PjRtClient> wrapped) : wrapped_(std::move(wrapped)) {}
    ~XlaMpiPjRtClient() override = default;

    int process_index() const override { return wrapped_->process_index(); }
    int device_count() const override { return wrapped_->device_count(); }
    int addressable_device_count() const override { return wrapped_->addressable_device_count(); }
    absl::Span<xla::PjRtDevice* const> devices() const override { return wrapped_->devices(); }
    absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
        return wrapped_->addressable_devices();
    }
    absl::StatusOr<xla::PjRtDevice*> LookupDevice(xla::GlobalDeviceId global_device_id) const override {
        return wrapped_->LookupDevice(global_device_id);
    }
    absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
        xla::LocalDeviceId local_device_id) const override {
        return wrapped_->LookupAddressableDevice(local_device_id);
    }
    void UpdateGlobalProcessInfo(absl::Span<xla::coordination::TaskInfo> infos) override {
        wrapped_->UpdateGlobalProcessInfo(infos);
    }
    absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
        return wrapped_->memory_spaces();
    }
    xla::PjRtPlatformId platform_id() const override { return wrapped_->platform_id(); }
    absl::string_view platform_name() const override { return wrapped_->platform_name(); }
    absl::string_view platform_version() const override { return wrapped_->platform_version(); }
    absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>> RuntimeAbiVersion() const override {
        return wrapped_->RuntimeAbiVersion();
    }
    std::optional<std::shared_ptr<xla::KeyValueStoreInterface>> key_value_store() const override {
        return wrapped_->key_value_store();
    }
    std::optional<xla::PjRtPluginAttributes> plugin_attributes() const override {
        return wrapped_->plugin_attributes();
    }
    absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(int num_replicas,
                                                                     int num_partitions) const override {
        return wrapped_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
    }
    absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
        int num_replicas, std::optional<int> num_replicas_per_slice, int num_partitions,
        const xla::MultiSliceConfig* multi_slice_config) const override {
        return wrapped_->GetDefaultDeviceAssignment(num_replicas, num_replicas_per_slice, num_partitions,
                                                    multi_slice_config);
    }
    absl::StatusOr<xla::Layout> GetDefaultLayout(xla::PrimitiveType element_type,
                                                 absl::Span<const int64_t> dims) override {
        return wrapped_->GetDefaultLayout(element_type, dims);
    }
    absl::StatusOr<std::unique_ptr<xla::HloCostAnalysis>> GetHloCostAnalysis() const override {
        return wrapped_->GetHloCostAnalysis();
    }

    absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(const xla::XlaComputation& computation,
                                                                 xla::CompileOptions options) override {
        return wrapped_->Compile(computation, std::move(options));
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
        const xla::XlaComputation& computation, xla::CompileOptions options) override {
        return wrapped_->CompileAndLoad(computation, std::move(options));
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(xla::MaybeOwningMlirModule module,
                                                                 xla::CompileOptions options) override;
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
        xla::MaybeOwningMlirModule module, xla::CompileOptions options) override;

    absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> DeserializeExecutable(
        absl::string_view serialized, std::optional<xla::CompileOptions> options) override {
        return wrapped_->DeserializeExecutable(serialized, std::move(options));
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> DeserializeExecutable(
        const absl::Cord& serialized, std::optional<xla::CompileOptions> options) override {
        return wrapped_->DeserializeExecutable(serialized, std::move(options));
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> LoadSerializedExecutable(
        absl::string_view serialized, std::optional<xla::CompileOptions> options,
        const xla::LoadOptions& load_options) override {
        return wrapped_->LoadSerializedExecutable(serialized, std::move(options), load_options);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> LoadSerializedExecutable(
        const absl::Cord& serialized, std::optional<xla::CompileOptions> options,
        const xla::LoadOptions& load_options) override {
        return wrapped_->LoadSerializedExecutable(serialized, std::move(options), load_options);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> Load(
        std::shared_ptr<xla::PjRtExecutable> executable, const xla::LoadOptions& load_options) override {
        return wrapped_->Load(std::move(executable), load_options);
    }

    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateUninitializedBuffer(
        const xla::Shape& shape, xla::PjRtMemorySpace* memory_space) override {
        return wrapped_->CreateUninitializedBuffer(shape, memory_space);
    }
    absl::StatusOr<std::pair<std::unique_ptr<xla::PjRtBuffer>, xla::PjRtFulfillAliasBufferCallback>>
    CreateAliasBuffer(const xla::Shape& shape, xla::PjRtMemorySpace* memory_space) override {
        return wrapped_->CreateAliasBuffer(shape, memory_space);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateErrorBuffer(
        absl::Status error, const xla::Shape& shape, xla::PjRtMemorySpace* memory) override {
        return wrapped_->CreateErrorBuffer(std::move(error), shape, memory);
    }
    absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription() const override {
        return wrapped_->GetTopologyDescription();
    }
    xla::HostMemoryAllocator* GetHostMemoryAllocator() const override {
        return wrapped_->GetHostMemoryAllocator();
    }

    absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>> CreateBuffersForAsyncHostToDevice(
        absl::Span<const ShapeSpec> shape_specs,
        std::optional<absl::Span<const std::optional<xla::Layout>>> device_layouts,
        xla::PjRtMemorySpace* memory_space) override {
        return wrapped_->CreateBuffersForAsyncHostToDevice(shape_specs, device_layouts, memory_space);
    }
    absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>> CreateBuffersForAsyncHostToDevice(
        absl::Span<const xla::Shape> shapes, xla::PjRtMemorySpace* memory_space) override {
        return wrapped_->CreateBuffersForAsyncHostToDevice(shapes, memory_space);
    }

    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
        const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
        std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
        absl::AnyInvocable<void() &&> on_done_with_host_buffer, xla::PjRtMemorySpace* memory_space,
        const xla::Layout* device_layout) override {
        return wrapped_->BufferFromHostBuffer(data, type, dims, byte_strides, host_buffer_semantics,
                                              std::move(on_done_with_host_buffer), memory_space,
                                              device_layout);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
        const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
        std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
        absl::AnyInvocable<void() &&> on_done_with_host_buffer, xla::PjRtBuffer* donated_dst,
        const xla::Layout* device_layout) override {
        return wrapped_->BufferFromHostBuffer(data, type, dims, byte_strides, host_buffer_semantics,
                                              std::move(on_done_with_host_buffer), donated_dst,
                                              device_layout);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostLiteral(
        const xla::LiteralSlice& literal, xla::PjRtMemorySpace* memory_space) override {
        return wrapped_->BufferFromHostLiteral(literal, memory_space);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostLiteral(
        const xla::LiteralSlice& literal, xla::PjRtMemorySpace* memory_space,
        const xla::Layout* device_layout) override {
        return wrapped_->BufferFromHostLiteral(literal, memory_space, device_layout);
    }
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateViewOfDeviceBuffer(
        void* device_ptr, const xla::Shape& shape, xla::PjRtMemorySpace* memory_space,
        std::function<void()> on_delete_callback,
        std::optional<std::intptr_t> stream = std::nullopt) override {
        return wrapped_->CreateViewOfDeviceBuffer(device_ptr, shape, memory_space,
                                                  std::move(on_delete_callback), stream);
    }
    absl::StatusOr<std::uintptr_t> UnsafeBufferPointer(xla::PjRtBuffer* buffer) override {
        return wrapped_->UnsafeBufferPointer(buffer);
    }
    absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> MakeCrossHostReceiveBuffers(
        absl::Span<const xla::Shape> shapes, xla::PjRtDevice* device,
        xla::PjRtCrossHostRecvNotifier notifier) override {
        return wrapped_->MakeCrossHostReceiveBuffers(shapes, device, std::move(notifier));
    }
    xla::PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager() const override {
        return wrapped_->GetPjRtHostMemoryForDeviceManager();
    }
    absl::Status DmaMap(void* data, size_t size) override { return wrapped_->DmaMap(data, size); }
    absl::Status DmaUnmap(void* data) override { return wrapped_->DmaUnmap(data); }
    absl::StatusOr<std::vector<xla::Future<>>> CrossHostSendBuffers(
        absl::Span<xla::PjRtBuffer* const> buffers,
        absl::Span<const xla::GlobalDeviceId> dst_global_device_ids,
        std::vector<xla::CrossHostTransferKey> transfer_keys) override {
        return wrapped_->CrossHostSendBuffers(buffers, dst_global_device_ids, std::move(transfer_keys));
    }
    absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> CrossHostReceiveBuffers(
        xla::PjRtDevice* device, absl::Span<const xla::Shape> shapes,
        absl::Span<const xla::GlobalDeviceId> src_global_device_ids,
        std::vector<xla::CrossHostTransferKey> transfer_keys) override {
        return wrapped_->CrossHostReceiveBuffers(device, shapes, src_global_device_ids,
                                                 std::move(transfer_keys));
    }

private:
    std::unique_ptr<xla::PjRtClient> wrapped_;
};

}  // namespace xla_mpi

#endif  // XLA_MPI_XLA_MPI_PJRT_CLIENT_H_
