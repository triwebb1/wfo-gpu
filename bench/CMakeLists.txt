add_executable(wfo_gpu_bench bench.cpp)
set_target_properties(wfo_gpu_bench PROPERTIES
    OUTPUT_NAME wfo-gpu-bench
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_link_libraries(wfo_gpu_bench PRIVATE wfo_gpu::core wfo_gpu_warnings)
