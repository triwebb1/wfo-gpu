#pragma once
// ---------------------------------------------------------------------------
// CLContext -- OpenCL context management with conditional compilation
//
// When HAS_OPENCL is defined (OpenCL SDK available), this wraps real OpenCL
// handles (platform, device, context, command queue).  When not defined,
// initialize() always returns false and is_available() is false, directing
// all compute through the CPU fallback path.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

// AUTO tries an OpenCL GPU, then an OpenCL CPU device. NATIVE skips OpenCL
// entirely and runs the multithreaded C++ path.
enum class DeviceType { GPU, CPU, AUTO, NATIVE };

class CLContext {
public:
    CLContext() = default;
    ~CLContext();

    // Non-copyable
    CLContext(const CLContext&)            = delete;
    CLContext& operator=(const CLContext&) = delete;

    // Movable
    CLContext(CLContext&& other) noexcept;
    CLContext& operator=(CLContext&& other) noexcept;

    // Attempt to initialize OpenCL.  Returns true if a device was found.
    // When HAS_OPENCL is not defined, always returns false.
    bool initialize(DeviceType preferred = DeviceType::AUTO);

    // Process-wide device preference used by the pipeline (screen,
    // walk-forward, detail). The CLI sets it from the [compute] config.
    static DeviceType  default_device();
    static void        set_default_device(DeviceType device);
    static const char* device_type_name(DeviceType device);

    // True if initialize() succeeded and OpenCL is usable.
    bool is_available() const { return m_available; }

    // Human-readable device name (empty when not available)
    std::string device_name() const { return m_device_name; }

    // Device global memory in bytes (0 when not available)
    size_t device_memory_bytes() const { return m_device_memory; }

    // Largest single buffer the device can allocate (0 when unavailable).
    size_t max_alloc_bytes() const { return m_max_alloc; }

    // Available (free) device memory in bytes.  Queries vendor-specific
    // extensions (NVIDIA CL_NV) at call time so the value reflects current
    // GPU state.  Falls back to 70% of total device memory if no extension
    // is available or the query fails.
    size_t available_memory_bytes() const;

    // Maximum work-group size (0 when not available)
    size_t max_work_group_size() const { return m_max_work_group_size; }

    // Release all OpenCL resources
    void release();

#ifdef HAS_OPENCL
    // Accessors for OpenCL handles (only available when SDK present)
    cl_platform_id   platform() const { return m_platform; }
    cl_device_id     device() const { return m_device; }
    cl_context       context() const { return m_context; }
    cl_command_queue queue() const { return m_queue; }
#endif

private:
    bool        m_available = false;
    std::string m_device_name;
    size_t      m_device_memory       = 0;
    size_t      m_max_alloc           = 0;
    size_t      m_max_work_group_size = 0;

#ifdef HAS_OPENCL
    cl_platform_id   m_platform = nullptr;
    cl_device_id     m_device   = nullptr;
    cl_context       m_context  = nullptr;
    cl_command_queue m_queue    = nullptr;
#endif
};
