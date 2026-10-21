#pragma once
// ---------------------------------------------------------------------------
// results_serde -- shared field-by-field serialization for BacktestResult.
//
// Single source of truth between checkpoint.cpp and
// orchestrator::write_results (all_results.bin). A raw memcpy of the
// BacktestResult struct would be ABI-fragile (sizeof, padding, compiler
// differences), so this module serializes each field explicitly to match
// the checkpoint format.
//
// Two surface APIs:
//   * write_backtest_result / read_backtest_result -- bare field emission,
//     used inside the checkpoint body (no length prefix here so the existing
//     checkpoint layout is byte-exact).
//   * write_all_results_file / read_all_results_file -- self-describing
//     all_results.bin writer: magic + version + count + per-record.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "metrics.h"

class DurableFileWriter;

namespace results_serde {

// All_results.bin format constants.
constexpr char     ALL_RESULTS_MAGIC[4] = {'W', 'F', 'A', 'R'};
constexpr uint32_t ALL_RESULTS_VERSION  = 1;

// ---- Streaming writers (used inside DurableFileWriter callbacks) ---------

void write_backtest_result(DurableFileWriter& w, const BacktestResult& r);

// Length-prefixed all_results.bin writer (magic+version+count+records).
void write_all_results_file(DurableFileWriter& w, const std::vector<BacktestResult>& results);

// ---- Stream-based helpers used by checkpoint.cpp read path ---------------
//
// The checkpoint reader uses std::ifstream; provide symmetric helpers that
// read one BacktestResult at a time from an istream.

bool read_backtest_result(std::istream& in, BacktestResult& r);

}  // namespace results_serde
