#include "pjrt_plugin/xla_mpi_pjrt_api.h"
#include "pjrt_plugin/mpi_collectives.h"
#include "pjrt_plugin/mpi_process_group.h"
#include "pjrt_plugin/xla_mpi_pjrt_client.h"

#include <mpi.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

namespace xla_mpi {

namespace {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
    PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
        "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE, args->struct_size));

    GetMpiSingleton().Init();

    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<xla::CpuTopology::CpuDevice> devices;
    devices.reserve(static_cast<size_t>(world_size));
    for (int p = 0; p < world_size; ++p) {
        devices.push_back(xla::CpuTopology::CpuDevice{p, /*local_device_id=*/0});
    }
    xla::CpuTopology cpu_topology(std::move(devices), xla::cpu::TargetMachineOptions());
    xla::CpuTopologyDescription topology(xla::CpuPlatformId(), xla::CpuPlatformName(),
                                         xla::CpuPlatformVersion(), cpu_topology);

    xla::CpuClientOptions options;
    options.process_id = rank;
    options.cpu_device_count = world_size;
    options.collectives = std::shared_ptr<xla::cpu::CpuCollectives>(
        &GetMpiSingleton().collectives(), [](xla::cpu::CpuCollectives*) {});
    options.topology = &topology;

    PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> real_client,
                          xla::GetPjRtCpuClient(std::move(options)));
    auto wrapped = std::make_unique<XlaMpiPjRtClient>(std::move(real_client));
    args->client = pjrt::CreateWrapperClient(GetXlaMpiPjrtApi(), std::move(wrapped));
    return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
    PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
        "PJRT_ExecuteContext_Create_Args", PJRT_ExecuteContext_Create_Args_STRUCT_SIZE,
        args->struct_size));
    auto execute_context = std::make_unique<xla::ExecuteContext>();
    args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
    return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args* args) {
    return pjrt::StatusToPjRtError(
        absl::UnimplementedError("Topology not supported for CPU compilation."));
}

PJRT_Error* PJRT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
    GetMpiSingleton().Init();
    return nullptr;
}

}  // namespace

const PJRT_Api* GetXlaMpiPjrtApi() {
    static const PJRT_Api pjrt_api =
        pjrt::CreatePjrtApi(xla_mpi::PJRT_Client_Create, xla_mpi::PJRT_ExecuteContext_Create,
                            xla_mpi::PJRT_TopologyDescription_Create, xla_mpi::PJRT_Plugin_Initialize);
    return &pjrt_api;
}

}  // namespace xla_mpi
