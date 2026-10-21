#pragma once
// The committed test fixture: three trading days of synthetic 30-second bars.
//
// tests/data/synthetic_30s.bin is generated from fixture_config() and
// tests/reference/synthetic_indicators_reference.csv is computed from it by
// the independent NumPy implementation in tools/gen_reference_indicators.py.
// tools/make_fixtures.py rebuilds both.

#include <cstdint>
#include <string>

#include "synthetic.h"

#ifndef WFO_GPU_TEST_DATA_DIR
#error "WFO_GPU_TEST_DATA_DIR must be defined by the build"
#endif
#ifndef WFO_GPU_TEST_REFERENCE_DIR
#error "WFO_GPU_TEST_REFERENCE_DIR must be defined by the build"
#endif

namespace test_fixture {

inline SyntheticConfig fixture_config() {
    SyntheticConfig cfg;
    cfg.seed          = 20240108;
    cfg.symbol        = "SYN";
    cfg.timeframe_sec = 30;
    cfg.calendar      = SyntheticCalendar::US_FUTURES;
    cfg.start_year    = 2024;
    cfg.start_month   = 1;
    cfg.start_day     = 8;
    cfg.num_days      = 3;
    return cfg;
}

// FNV-1a 64 over the generated Bar array (see test_synthetic.cpp).
inline constexpr uint64_t FIXTURE_BARS_FNV1A = 0x7f244c1afb3e054eULL;

inline std::string fixture_path() {
    return std::string(WFO_GPU_TEST_DATA_DIR) + "/synthetic_30s.bin";
}

inline std::string reference_csv_path() {
    return std::string(WFO_GPU_TEST_REFERENCE_DIR) + "/synthetic_indicators_reference.csv";
}

}  // namespace test_fixture
