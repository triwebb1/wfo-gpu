function(wfo_gpu_example target output source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${output}
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    target_link_libraries(${target} PRIVATE wfo_gpu::core wfo_gpu_warnings)
endfunction()

wfo_gpu_example(wfo_gpu_example_custom_signal wfo-gpu-example-custom-signal
                custom_signal/main.cpp)
wfo_gpu_example(wfo_gpu_example_walk_forward wfo-gpu-example-walk-forward
                walk_forward_loop/main.cpp)
