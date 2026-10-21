#pragma once
// ---------------------------------------------------------------------------
// CLMemory -- Device memory management with SoA layout
//
// Manages aligned host buffers and (when available) OpenCL device buffers
// for bar data in Structure-of-Arrays format.  Provides upload/download
// operations for bar data, indicator arrays, and result buffers.
//
// When OpenCL is not available, the SoA host buffers are used directly
// by the CPU fallback path.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cl_context.h"
#include "../market_data.h"

// Required alignment for all compute buffers (device cache line)
constexpr size_t CL_BUFFER_ALIGNMENT = 128;

// ---------------------------------------------------------------------------
// SoA bar data layout -- separate contiguous arrays per field
// ---------------------------------------------------------------------------
struct SoABarData {
    std::vector<float>    open;
    std::vector<float>    high;
    std::vector<float>    low;
    std::vector<float>    close;
    std::vector<uint32_t> volume;
    int                   num_bars = 0;

    // Convert AoS Bar array to SoA
    void from_bars(const Bar* bars, int start, int count);

    // Clear all arrays
    void clear();
};

// ---------------------------------------------------------------------------
// CLMemory -- manages host + device buffers
// ---------------------------------------------------------------------------
class CLMemory {
public:
    CLMemory() = default;
    ~CLMemory();

    // Non-copyable
    CLMemory(const CLMemory&)            = delete;
    CLMemory& operator=(const CLMemory&) = delete;

    // ---- SoA bar data ----

    // Convert bars[start .. start+count) to SoA layout.
    // Returns true on success.
    bool upload_bars(const CLContext& ctx, const Bar* bars, int start, int count);

    // Access host-side SoA data (always available, even without OpenCL)
    const SoABarData& soa_bars() const { return m_soa; }

    // ---- Indicator output arrays ----

    // Allocate N indicator output arrays, each of length num_bars floats.
    // Returns pointers to the host-side arrays.
    std::vector<float*> allocate_indicator_buffers(const CLContext& ctx, int num_series,
                                                   int num_bars);

    // ---- Result buffers ----

    // Allocate a result buffer of `count` elements, each `element_bytes` in size.
    // Returns pointer to host-side buffer.
    void* allocate_result_buffer(const CLContext& ctx, int count, size_t element_bytes);

    // Download results from device to host (no-op if no OpenCL)
    bool download_results(const CLContext& ctx);

    // Access raw result buffer
    void*       result_data() { return m_result_host.data(); }
    const void* result_data() const { return m_result_host.data(); }

    // ---- Cleanup ----
    void release_all();

#ifdef HAS_OPENCL
    cl_mem dev_result_buffer() const { return m_dev_result; }
#endif

private:
    // Host-side SoA bar data
    SoABarData m_soa;

    // Host-side indicator arrays (contiguous block, subdivided into series)
    std::vector<float>  m_indicator_host;
    std::vector<float*> m_indicator_ptrs;
    int                 m_indicator_num_series = 0;
    int                 m_indicator_num_bars   = 0;

    // Host-side result buffer
    std::vector<uint8_t> m_result_host;
    int                  m_result_count        = 0;
    size_t               m_result_element_size = 0;

#ifdef HAS_OPENCL
    cl_mem m_dev_result = nullptr;

    void release_device_buffers();
#endif

    // Aligned allocation helper
    static size_t aligned_size(size_t bytes);
};
