#include "cl_context.h"

#include <atomic>
#include <cstring>
#include <iostream>

// Suppress deprecation warning for clCreateCommandQueue (OpenCL 1.x compatible)
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

// NVIDIA vendor extension: free device memory in KB (two-element array)
#ifndef CL_DEVICE_GPU_MEMORY_FREE_NV
#define CL_DEVICE_GPU_MEMORY_FREE_NV 0x4039
#endif

// ---------------------------------------------------------------------------
// Destructor / Move / Release
// ---------------------------------------------------------------------------

CLContext::~CLContext() {
    release();
}

CLContext::CLContext(CLContext&& other) noexcept {
    *this = std::move(other);
}

CLContext& CLContext::operator=(CLContext&& other) noexcept {
    if (this != &other) {
        release();
        m_available           = other.m_available;
        m_device_name         = std::move(other.m_device_name);
        m_device_memory       = other.m_device_memory;
        m_max_alloc           = other.m_max_alloc;
        m_max_work_group_size = other.m_max_work_group_size;

#ifdef HAS_OPENCL
        m_platform = other.m_platform;
        m_device   = other.m_device;
        m_context  = other.m_context;
        m_queue    = other.m_queue;

        other.m_platform = nullptr;
        other.m_device   = nullptr;
        other.m_context  = nullptr;
        other.m_queue    = nullptr;
#endif

        other.m_available           = false;
        other.m_device_memory       = 0;
        other.m_max_alloc           = 0;
        other.m_max_work_group_size = 0;
    }
    return *this;
}

void CLContext::release() {
#ifdef HAS_OPENCL
    if (m_queue) {
        clFlush(m_queue);
        clFinish(m_queue);
        clReleaseCommandQueue(m_queue);
        m_queue = nullptr;
    }
    if (m_context) {
        clReleaseContext(m_context);
        m_context = nullptr;
    }
    // device and platform are not retained/released in OpenCL 1.2
    m_device   = nullptr;
    m_platform = nullptr;
#endif
    m_available = false;
    m_device_name.clear();
    m_device_memory       = 0;
    m_max_alloc           = 0;
    m_max_work_group_size = 0;
}

// ---------------------------------------------------------------------------
// available_memory_bytes() -- query free GPU memory at call time
// ---------------------------------------------------------------------------

size_t CLContext::available_memory_bytes() const {
    if (!m_available || m_device_memory == 0) return 0;

    // Only print the diagnostic message once per process
    static bool s_printed = false;

#ifdef HAS_OPENCL
    // NVIDIA: CL_DEVICE_GPU_MEMORY_FREE_NV returns free memory in KB
    {
        cl_ulong free_kb[2] = {0, 0};
        cl_int   err = clGetDeviceInfo(m_device, CL_DEVICE_GPU_MEMORY_FREE_NV, sizeof(free_kb),
                                       free_kb, nullptr);
        if (err == CL_SUCCESS && free_kb[0] > 0) {
            size_t free_bytes = static_cast<size_t>(free_kb[0]) * 1024u;
            if (!s_printed) {
                std::cout << "[CLContext] GPU free memory: " << (free_bytes / (1024 * 1024))
                          << " MB / " << (m_device_memory / (1024 * 1024)) << " MB total"
                          << std::endl;
                s_printed = true;
            }
            return free_bytes;
        }
    }
#endif

    // Fallback: no vendor extension available.  Assume 70% of total is usable
    // (conservative estimate accounting for driver + other processes).
    size_t fallback = static_cast<size_t>(static_cast<double>(m_device_memory) * 0.70);
    if (!s_printed) {
        std::cout << "[CLContext] GPU free memory query not supported; "
                  << "using 70% of " << (m_device_memory / (1024 * 1024))
                  << " MB = " << (fallback / (1024 * 1024)) << " MB" << std::endl;
        s_printed = true;
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// initialize()
// ---------------------------------------------------------------------------

namespace {
std::atomic<DeviceType> g_default_device{DeviceType::AUTO};
}

DeviceType CLContext::default_device() {
    return g_default_device.load();
}

void CLContext::set_default_device(DeviceType device) {
    g_default_device.store(device);
}

const char* CLContext::device_type_name(DeviceType device) {
    switch (device) {
        case DeviceType::GPU:
            return "gpu";
        case DeviceType::CPU:
            return "opencl-cpu";
        case DeviceType::AUTO:
            return "auto";
        case DeviceType::NATIVE:
            return "native";
    }
    return "?";
}

bool CLContext::initialize(DeviceType preferred) {
    release();
    if (preferred == DeviceType::NATIVE) return false;

#ifdef HAS_OPENCL
    // ----- Find platforms -----
    cl_uint num_platforms = 0;
    cl_int  err           = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) return false;

    // ----- Search for a device -----
    // Order of preference based on `preferred`:
    //   AUTO -> try GPU first, then CPU
    //   GPU  -> GPU only
    //   CPU  -> CPU only
    auto try_device_type = [&](cl_device_type cl_type) -> bool {
        for (auto& plat : platforms) {
            cl_uint num_devices = 0;
            err                 = clGetDeviceIDs(plat, cl_type, 0, nullptr, &num_devices);
            if (err != CL_SUCCESS || num_devices == 0) continue;

            std::vector<cl_device_id> devices(num_devices);
            err = clGetDeviceIDs(plat, cl_type, num_devices, devices.data(), nullptr);
            if (err != CL_SUCCESS) continue;

            // Take the first device
            m_platform = plat;
            m_device   = devices[0];

            // Query device name
            char name_buf[256] = {};
            clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);
            m_device_name = name_buf;

            // Query device memory
            cl_ulong mem_size = 0;
            clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size,
                            nullptr);
            m_device_memory = static_cast<size_t>(mem_size);

            cl_ulong max_alloc = 0;
            clGetDeviceInfo(m_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc,
                            nullptr);
            m_max_alloc = static_cast<size_t>(max_alloc);

            // Query max work-group size
            size_t wg_size = 0;
            clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg_size), &wg_size,
                            nullptr);
            m_max_work_group_size = wg_size;

            // Create context
            m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                m_context = nullptr;
                continue;
            }

            // Create command queue.
            // properties=0 selects an IN-ORDER queue (the default).
            // Downstream indicator and backtest kernels rely on in-order
            // semantics: inputs written via clEnqueueWriteBuffer must commit
            // before kernel launch, and successive kernels that share
            // m_pool_params / m_pool_output must serialize implicitly.
            // Do NOT enable CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE without
            // adding explicit event-based synchronization.
            m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
            if (err != CL_SUCCESS) {
                clReleaseContext(m_context);
                m_context = nullptr;
                m_queue   = nullptr;
                continue;
            }

            m_available = true;
            return true;
        }
        return false;
    };

    switch (preferred) {
        case DeviceType::GPU:
            return try_device_type(CL_DEVICE_TYPE_GPU);

        case DeviceType::CPU:
            return try_device_type(CL_DEVICE_TYPE_CPU);

        case DeviceType::AUTO:
        default:
            if (try_device_type(CL_DEVICE_TYPE_GPU)) return true;
            return try_device_type(CL_DEVICE_TYPE_CPU);
    }

#else
    // No OpenCL SDK -- always fall back to CPU path
    (void)preferred;
    return false;
#endif
}
