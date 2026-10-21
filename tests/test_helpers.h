#pragma once
// Shared test helpers -- eliminates make_bar() duplication across 12 test files.

#include <cstdint>
#include <cstdlib>
#include <vector>
#include "market_data.h"
#include "cost_model.h"
#include "backtest_engine.h"

static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

inline Bar make_bar(int64_t ts, float o, float h, float l, float c, uint32_t v = 100) {
    Bar b{};
    b.timestamp = ts;
    b.open      = o;
    b.high      = h;
    b.low       = l;
    b.close     = c;
    b.volume    = v;
    b._pad      = 0;
    return b;
}

// Skip a test that needs an OpenCL device when none is available. With the
// WFO_GPU_REQUIRE_OPENCL environment variable set (CI's OpenCL job) a
// missing device fails the test instead, so a broken runtime cannot pass
// as a run of skipped tests.
#define SKIP_UNLESS_OPENCL(reason)                                    \
    do {                                                              \
        if (std::getenv("WFO_GPU_REQUIRE_OPENCL") != nullptr) {       \
            FAIL() << "WFO_GPU_REQUIRE_OPENCL is set but " << reason; \
        }                                                             \
        GTEST_SKIP() << reason;                                       \
    } while (0)

inline CostModel es_cost() {
    return CostModel::es();
}

inline std::vector<Bar> make_daily_bars(int N, float base_price = 4500.0f) {
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, base_price, base_price + 1.0f, base_price - 1.0f, base_price);
    }
    return bars;
}
