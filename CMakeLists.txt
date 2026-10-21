cmake_minimum_required(VERSION 3.20)
project(wfo_gpu
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated walk-forward optimization for trading strategies"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(WFO_GPU_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(WFO_GPU_IS_TOP_LEVEL ON)
endif()

option(WFO_GPU_ENABLE_OPENCL      "Build the OpenCL backend if an OpenCL SDK is found" ON)
option(WFO_GPU_BUILD_TESTS        "Build the test suite"            ${WFO_GPU_IS_TOP_LEVEL})
option(WFO_GPU_BUILD_BENCH        "Build the benchmark tool"        ${WFO_GPU_IS_TOP_LEVEL})
option(WFO_GPU_BUILD_EXAMPLES     "Build the examples"              ${WFO_GPU_IS_TOP_LEVEL})
option(WFO_GPU_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(WFO_GPU_SANITIZE           "Build with AddressSanitizer and UBSan (GCC/Clang)" OFF)

find_package(Threads REQUIRED)

include(FetchContent)
# toml++ (header-only, MIT) parses run configs.
FetchContent_Declare(tomlplusplus
    URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(tomlplusplus)

# ---- Warnings and sanitizers (applied to project targets only) ----
add_library(wfo_gpu_warnings INTERFACE)
if(MSVC)
    target_compile_options(wfo_gpu_warnings INTERFACE /W4 /permissive- /utf-8)
    target_compile_definitions(wfo_gpu_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
    if(WFO_GPU_WARNINGS_AS_ERRORS)
        target_compile_options(wfo_gpu_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(wfo_gpu_warnings INTERFACE -Wall -Wextra -Wpedantic)
    if(WFO_GPU_WARNINGS_AS_ERRORS)
        target_compile_options(wfo_gpu_warnings INTERFACE -Werror)
    endif()
    if(WFO_GPU_SANITIZE)
        target_compile_options(wfo_gpu_warnings INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
        target_link_options(wfo_gpu_warnings INTERFACE -fsanitize=address,undefined)
    endif()
endif()

# ---- OpenCL (optional) ----
set(WFO_GPU_HAS_OPENCL OFF)
if(WFO_GPU_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(WFO_GPU_HAS_OPENCL ON)
        message(STATUS "wfo-gpu: OpenCL backend enabled (${OpenCL_LIBRARIES})")
    else()
        message(STATUS "wfo-gpu: OpenCL SDK not found; building the CPU backend only")
    endif()
endif()

# ---- Embed OpenCL kernels as generated headers ----
# Kernels are compiled into the binary so the GPU path never depends on the
# working directory or on files shipped next to the executable.
set(WFO_GPU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
function(wfo_gpu_embed_kernel name symbol)
    set(in  "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/kernels/${name}.cl")
    set(out "${WFO_GPU_GENERATED_DIR}/${name}_cl_source.h")
    add_custom_command(
        OUTPUT  ${out}
        DEPENDS ${in} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernel.cmake
        COMMAND ${CMAKE_COMMAND} -DIN=${in} -DOUT=${out} -DSYMBOL=${symbol}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernel.cmake
        VERBATIM
        COMMENT "Embedding ${name}.cl")
    set(WFO_GPU_KERNEL_HEADERS ${WFO_GPU_KERNEL_HEADERS} ${out} PARENT_SCOPE)
endfunction()
wfo_gpu_embed_kernel(indicators INDICATORS_CL_SOURCE)
wfo_gpu_embed_kernel(backtest   BACKTEST_CL_SOURCE)

# ---- Core library ----
set(WFO_GPU_CORE_SOURCES
    src/backtest_engine.cpp
    src/bar_io.cpp
    src/checkpoint.cpp
    src/config.cpp
    src/crc32c.cpp
    src/detail_runner.cpp
    src/durable_file.cpp
    src/landscape.cpp
    src/market_data.cpp
    src/metrics.cpp
    src/orchestrator.cpp
    src/pareto.cpp
    src/pnl_decomposition.cpp
    src/resource_monitor.cpp
    src/results.cpp
    src/results_serde.cpp
    src/robustness.cpp
    src/run_lock.cpp
    src/selection.cpp
    src/session_model.cpp
    src/synthetic.cpp
    src/wfo_engine.cpp

    src/compute/cl_backtest.cpp
    src/compute/cl_context.cpp
    src/compute/cl_indicators.cpp
    src/compute/cl_memory.cpp
    src/compute/indicator_warmup.cpp

    src/grids/indicator_grid.cpp
    src/grids/session_signal_grid.cpp
    src/grids/strategy_grid.cpp
    src/grids/wfo_grid.cpp
    src/grids/window_mode_grid.cpp

    src/indicators/adx.cpp
    src/indicators/atr.cpp
    src/indicators/bollinger.cpp
    src/indicators/cci.cpp
    src/indicators/connors_rsi.cpp
    src/indicators/donchian.cpp
    src/indicators/ema_sma_cross.cpp
    src/indicators/hma.cpp
    src/indicators/keltner.cpp
    src/indicators/macd.cpp
    src/indicators/obv.cpp
    src/indicators/pivot_points.cpp
    src/indicators/roc.cpp
    src/indicators/rsi.cpp
    src/indicators/signal_normalize.cpp
    src/indicators/stochastic.cpp
    src/indicators/vwap.cpp

    src/session_signals/calendar.cpp
    src/session_signals/gap_analysis.cpp
    src/session_signals/intraday_structure.cpp
    src/session_signals/level_interaction.cpp
    src/session_signals/multiday_patterns.cpp
    src/session_signals/opening_range.cpp
    src/session_signals/overnight_context.cpp
    src/session_signals/volatility_regime.cpp
)

add_library(wfo_gpu_core STATIC ${WFO_GPU_CORE_SOURCES} ${WFO_GPU_KERNEL_HEADERS})
add_library(wfo_gpu::core ALIAS wfo_gpu_core)
target_include_directories(wfo_gpu_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${WFO_GPU_GENERATED_DIR}")
target_link_libraries(wfo_gpu_core
    PUBLIC  Threads::Threads
    PRIVATE wfo_gpu_warnings tomlplusplus::tomlplusplus)
target_compile_features(wfo_gpu_core PUBLIC cxx_std_20)
if(WFO_GPU_HAS_OPENCL)
    # Keep the OpenCL 1.2 entry points (clCreateCommandQueue) usable without
    # deprecation warnings, so the code runs on 1.2-only platforms too.
    target_compile_definitions(wfo_gpu_core PUBLIC
        HAS_OPENCL CL_TARGET_OPENCL_VERSION=300 CL_USE_DEPRECATED_OPENCL_1_2_APIS)
    target_link_libraries(wfo_gpu_core PUBLIC OpenCL::OpenCL)
endif()

# Runtime config (early-close calendar) is looked up relative to the working
# directory; stage a copy so the CLI and tests run from the build tree.
configure_file(config/cme_early_closes.csv
               "${CMAKE_CURRENT_BINARY_DIR}/config/cme_early_closes.csv" COPYONLY)

# ---- Command-line tool ----
add_executable(wfo_gpu_cli
    src/cli/main.cpp
    src/cli/data_commands.cpp
    src/cli/pipeline_commands.cpp)
set_target_properties(wfo_gpu_cli PROPERTIES OUTPUT_NAME wfo-gpu)
target_compile_definitions(wfo_gpu_cli PRIVATE WFO_GPU_VERSION="${PROJECT_VERSION}")
target_link_libraries(wfo_gpu_cli PRIVATE wfo_gpu::core wfo_gpu_warnings)

# ---- Tests ----
if(WFO_GPU_BUILD_TESTS)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(wfo_gpu_tests
        tests/test_backtest_edge.cpp
        tests/test_bar_io.cpp
        tests/test_checkpoint_resources.cpp
        tests/test_config.cpp
        tests/test_core.cpp
        tests/test_detail_runner.cpp
        tests/test_device_parity.cpp
        tests/test_edge_cases.cpp
        tests/test_grids.cpp
        tests/test_indicators_cross_validate.cpp
        tests/test_indicators_full.cpp
        tests/test_integration.cpp
        tests/test_lookahead.cpp
        tests/test_losers_trigger.cpp
        tests/test_main.cpp
        tests/test_metrics_edge.cpp
        tests/test_opencl.cpp
        tests/test_platform.cpp
        tests/test_resume.cpp
        tests/test_screen.cpp
        tests/test_session_close.cpp
        tests/test_session_integration.cpp
        tests/test_session_signals.cpp
        tests/test_signal_normalize.cpp
        tests/test_strategy_execution.cpp
        tests/test_synthetic.cpp
        tests/test_wfo.cpp
    )
    target_link_libraries(wfo_gpu_tests PRIVATE
        wfo_gpu::core wfo_gpu_warnings GTest::gtest)
    target_compile_definitions(wfo_gpu_tests PRIVATE
        WFO_GPU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        WFO_GPU_TEST_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/reference"
        WFO_GPU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    include(GoogleTest)
    gtest_discover_tests(wfo_gpu_tests
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DISCOVERY_TIMEOUT 60)
endif()

# ---- Benchmark and examples ----
if(WFO_GPU_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(WFO_GPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
