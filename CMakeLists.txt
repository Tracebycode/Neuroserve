cmake_minimum_required(VERSION 3.22)
project(neuroserve VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Strict compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
        -pthread
        -O3
    )
endif()

include(FetchContent)

# 1. spdlog
message(STATUS "Fetching spdlog...")
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

# 2. yaml-cpp
message(STATUS "Fetching yaml-cpp...")
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(yaml-cpp)

# 3. nlohmann/json (Single header release archive is instant)
message(STATUS "Fetching nlohmann_json...")
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# 4. cpp-httplib
message(STATUS "Fetching cpp-httplib...")
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(httplib)

# 5. ONNX Runtime Linux x64 Prebuilt
message(STATUS "Fetching ONNX Runtime prebuilt...")
FetchContent_Declare(
    onnxruntime_prebuilt
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
)
FetchContent_MakeAvailable(onnxruntime_prebuilt)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so.1.18.0"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include"
)

# 6. GoogleTest
message(STATUS "Fetching GoogleTest...")
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

# 7. Google Benchmark
message(STATUS "Fetching Google Benchmark...")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(benchmark)

# Core library
add_library(neuroserve_core
    src/config/config.cpp
    src/common/logger.cpp
    src/model/model_instance.cpp
    src/model/model_manager.cpp
    src/batching/dynamic_batcher.cpp
    src/worker/worker_pool.cpp
    src/metrics/metrics_registry.cpp
    src/inference/inference_service.cpp
    src/server/http_server.cpp
)

target_include_directories(neuroserve_core PUBLIC
    include
)

target_link_libraries(neuroserve_core PUBLIC
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    httplib::httplib
    onnxruntime
    pthread
)

# Set rpath so libonnxruntime.so is found at runtime
set_target_properties(neuroserve_core PROPERTIES
    BUILD_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
    INSTALL_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
)

# Main Server Executable
add_executable(neuroserve_server src/main.cpp)
target_link_libraries(neuroserve_server PRIVATE neuroserve_core)
set_target_properties(neuroserve_server PROPERTIES
    BUILD_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
    INSTALL_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
)

# Tests
enable_testing()
add_executable(neuroserve_tests
    tests/test_main.cpp
    tests/test_config.cpp
    tests/test_bounded_queue.cpp
    tests/test_model_manager.cpp
    tests/test_dynamic_batcher.cpp
    tests/test_worker_pool.cpp
    tests/test_load_shedding.cpp
    tests/test_inference_service.cpp
)
target_link_libraries(neuroserve_tests PRIVATE neuroserve_core GTest::gtest)
set_target_properties(neuroserve_tests PROPERTIES
    BUILD_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
    INSTALL_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
)
add_test(NAME AllUnitTests COMMAND neuroserve_tests)

# Benchmarks
add_executable(neuroserve_bench
    benchmarks/bench_main.cpp
    benchmarks/bench_queue.cpp
    benchmarks/bench_inference.cpp
)
target_link_libraries(neuroserve_bench PRIVATE neuroserve_core benchmark::benchmark)
set_target_properties(neuroserve_bench PROPERTIES
    BUILD_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
    INSTALL_RPATH "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
)
