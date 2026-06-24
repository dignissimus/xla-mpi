cmake_minimum_required(VERSION 3.20)
project(xlampi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
message(STATUS "Found MPI CXX compiler: ${MPI_CXX_COMPILER}")

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/xlampi-deps")

set(XLA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

set(PJRT_SOURCES
    src/pjrt_plugin/pjrt_api.cc
    src/pjrt_plugin/pjrt_types.cc
    src/pjrt_plugin/pjrt_error.cc
    src/pjrt_plugin/pjrt_client.cc
    src/pjrt_plugin/pjrt_device.cc
    src/pjrt_plugin/pjrt_topology.cc
    src/pjrt_plugin/pjrt_memory.cc
    src/pjrt_plugin/mpi_executable.cc
)

add_library(pjrt_plugin_mpi SHARED ${PJRT_SOURCES})

target_include_directories(pjrt_plugin_mpi PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${XLA_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(pjrt_plugin_mpi PRIVATE
    MPI::MPI_CXX
)

set_target_properties(pjrt_plugin_mpi PROPERTIES
    OUTPUT_NAME "pjrt_plugin_mpi"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
