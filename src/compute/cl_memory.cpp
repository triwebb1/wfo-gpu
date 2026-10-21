#include "cl_memory.h"

#include <algorithm>
#include <cassert>
#include <cstring>

// ---------------------------------------------------------------------------
// SoABarData
// ---------------------------------------------------------------------------

void SoABarData::from_bars(const Bar* bars, int start, int count) {
    assert(count > 0);
    num_bars = count;

    open.resize(static_cast<size_t>(count));
    high.resize(static_cast<size_t>(count));
    low.resize(static_cast<size_t>(count));
    close.resize(static_cast<size_t>(count));
    volume.resize(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i) {
        const Bar& b = bars[start + i];
        open[i]      = b.open;
        high[i]      = b.high;
        low[i]       = b.low;
        close[i]     = b.close;
        volume[i]    = b.volume;
    }
}

void SoABarData::clear() {
    open.clear();
    high.clear();
    low.clear();
    close.clear();
    volume.clear();
    num_bars = 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

size_t CLMemory::aligned_size(size_t bytes) {
    // Round up to next multiple of CL_BUFFER_ALIGNMENT
    return (bytes + CL_BUFFER_ALIGNMENT - 1) & ~(CL_BUFFER_ALIGNMENT - 1);
}

// ---------------------------------------------------------------------------
// Destructor / cleanup
// ---------------------------------------------------------------------------

CLMemory::~CLMemory() {
    release_all();
}

void CLMemory::release_all() {
#ifdef HAS_OPENCL
    release_device_buffers();
#endif

    m_soa.clear();
    m_indicator_host.clear();
    m_indicator_ptrs.clear();
    m_indicator_num_series = 0;
    m_indicator_num_bars   = 0;
    m_result_host.clear();
    m_result_count        = 0;
    m_result_element_size = 0;
}

#ifdef HAS_OPENCL
void CLMemory::release_device_buffers() {
    auto safe_release = [](cl_mem& mem) {
        if (mem) {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
    };
    safe_release(m_dev_result);
}
#endif

// ---------------------------------------------------------------------------
// upload_bars
// ---------------------------------------------------------------------------

bool CLMemory::upload_bars(const CLContext& ctx, const Bar* bars, int start, int count) {
    // Always populate host-side SoA
    m_soa.from_bars(bars, start, count);

    // No device bar buffers here -- CLIndicators manages its own pooled device
    // buffers, so CLMemory device copies would never be read.
    (void)ctx;

    return true;  // Host-side SoA is populated -- CPU fallback ready
}

// ---------------------------------------------------------------------------
// allocate_indicator_buffers
// ---------------------------------------------------------------------------

std::vector<float*> CLMemory::allocate_indicator_buffers(const CLContext& ctx, int num_series,
                                                         int num_bars) {
    m_indicator_num_series = num_series;
    m_indicator_num_bars   = num_bars;

    // Allocate one contiguous block: num_series * aligned_count floats
    const size_t aligned_count =
        aligned_size(static_cast<size_t>(num_bars) * sizeof(float)) / sizeof(float);
    const size_t total_floats = static_cast<size_t>(num_series) * aligned_count;

    m_indicator_host.resize(total_floats, 0.0f);
    m_indicator_ptrs.resize(static_cast<size_t>(num_series));

    for (int s = 0; s < num_series; ++s) {
        m_indicator_ptrs[s] = m_indicator_host.data() + static_cast<size_t>(s) * aligned_count;
    }

    // No device indicator buffers here -- CLIndicators manages its own pooled
    // device output buffers, so CLMemory device copies would never be written.
    (void)ctx;

    return m_indicator_ptrs;
}

// ---------------------------------------------------------------------------
// allocate_result_buffer
// ---------------------------------------------------------------------------

void* CLMemory::allocate_result_buffer(const CLContext& ctx, int count, size_t element_bytes) {
    m_result_count        = count;
    m_result_element_size = element_bytes;

    const size_t total = aligned_size(static_cast<size_t>(count) * element_bytes);
    m_result_host.resize(total, 0);

#ifdef HAS_OPENCL
    if (ctx.is_available()) {
        if (m_dev_result) {
            clReleaseMemObject(m_dev_result);
            m_dev_result = nullptr;
        }
        cl_int err;
        m_dev_result = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, total, nullptr, &err);
        if (err != CL_SUCCESS) {
            m_dev_result = nullptr;
        }
    }
#else
    (void)ctx;
#endif

    return m_result_host.data();
}

// ---------------------------------------------------------------------------
// download_results
// ---------------------------------------------------------------------------

bool CLMemory::download_results(const CLContext& ctx) {
#ifdef HAS_OPENCL
    if (ctx.is_available() && m_dev_result) {
        const size_t total = static_cast<size_t>(m_result_count) * m_result_element_size;
        cl_int       err   = clEnqueueReadBuffer(ctx.queue(), m_dev_result, CL_TRUE, 0, total,
                                                 m_result_host.data(), 0, nullptr, nullptr);
        return err == CL_SUCCESS;
    }
#else
    (void)ctx;
#endif
    return true;  // no-op for CPU path
}
