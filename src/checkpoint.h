#pragma once
// ---------------------------------------------------------------------------
// Checkpoint -- save/restore progress for long-running screen and walk-forward runs
//
// Saves binary checkpoint files to ./output/checkpoints/ with magic
// bytes and version number for validation.  Each checkpoint captures the
// current phase, how far it has progressed, and all completed results.
//
// File format (version 12):
//   bytes 0-3:    magic "WFCK"
//   bytes 4-7:    version (uint32_t)
//   bytes 8-15:   grid_layout_fingerprint (u64)
//   bytes 16-23:  config_hash           (u64)
//   bytes 24-31:  bar_data_hash         (u64)
//   bytes 32-39:  scan_start_bar        (i64)
//   bytes 40-47:  scan_end_bar          (i64)
//   bytes 48+:    serialized Checkpoint body
//   trailing 4 bytes: CRC32C of every byte written before it
//
// Integrity checks:
//   - Writes go through DurableFileWriter (CreateFileW + FlushFileBuffers +
//     MoveFileExW atomic rename), so a crash never leaves a partial file.
//   - config_hash (FNV-1a over every result-changing ScreenConfig /
//     WalkForwardConfig input, scan range included) and bar_data_hash
//     (FNV-1a over the bar payload) must match on load. scan_start_bar and
//     scan_end_bar are also stored in the header for inspection.
//   - grid_layout_fingerprint must match the running build's grid constants.
//   - Trailing CRC32C (Castagnoli) covers the whole file.
//   - Versions outside [CHECKPOINT_VERSION_MIN_COMPAT, CHECKPOINT_VERSION]
//     are rejected with no migration: session_day semantics and the
//     serialized fields (including PnLDecomposition::per_trade) must match
//     exactly, or resumed WFO window boundaries and metrics would desync.
//   - Any failure makes Checkpoint::load return std::nullopt.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "metrics.h"
#include "orchestrator.h"
#include "wfo_engine.h"

// Magic bytes for checkpoint file validation
constexpr char     CHECKPOINT_MAGIC[4]           = {'W', 'F', 'C', 'K'};
constexpr uint32_t CHECKPOINT_VERSION            = 12;  // current format
constexpr uint32_t CHECKPOINT_VERSION_MIN_COMPAT = 12;  // older versions rejected (no migration)

// Runtime fingerprint of the grid layout constants.
uint64_t compute_grid_layout_fingerprint();

// FNV-1a over every ScreenConfig / WalkForwardConfig input that changes
// results: scan range, floors, costs, entry lag, strategy budget and session
// model for the screen; costs, session model, out-of-sample start, bars per
// day and the reduced strategy set for the walk-forward. File paths are not
// hashed.
uint64_t compute_config_hash_screen(const Orchestrator::ScreenConfig& c);
uint64_t compute_config_hash_walk_forward(const Orchestrator::WalkForwardConfig& c);

struct Checkpoint {
    std::string run_id;
    int         phase                   = 0;  // 1 = screen, 2 = walk-forward
    uint64_t    grid_layout_fingerprint = 0;

    // Resume invariants -- stamped at save, re-checked at load.
    uint64_t config_hash    = 0;
    uint64_t bar_data_hash  = 0;
    int64_t  scan_start_bar = 0;
    int64_t  scan_end_bar   = 0;

    // Screen progress
    int                         strategies_evaluated = 0;
    int                         total_strategies     = 0;
    std::vector<BacktestResult> completed_results;

    // Walk-forward progress
    int                       current_schedule_idx = 0;
    int                       current_window_idx   = 0;
    std::vector<WFORunResult> completed_runs;

    // Save checkpoint to a binary file.
    // Creates parent directories if they do not exist.
    // Uses DurableFileWriter: crash-safe atomic rename + CRC32C.
    // Returns true on success.
    bool save(const std::string& path) const;

    // Load checkpoint from a binary file.
    // Returns std::nullopt on ANY failure (missing file, bad magic,
    // version out of compat band, fingerprint/config/data hash mismatch,
    // truncated/corrupt body, CRC mismatch).
    //
    // If expected_config_hash != 0, rejects when stored config_hash doesn't match.
    // If expected_bar_data_hash != 0, rejects when stored bar_data_hash doesn't match.
    // Pass 0 for both to skip those checks (used by is_valid / test harnesses).
    static std::optional<Checkpoint> load(const std::string& path,
                                          uint64_t           expected_config_hash   = 0,
                                          uint64_t           expected_bar_data_hash = 0);

    // Check if a checkpoint file exists and is parseable end-to-end.
    // Equivalent to load(path, 0, 0).has_value() -- single source of truth
    // for validity semantics (magic + version band + fingerprint + CRC).
    static bool is_valid(const std::string& path);

    // Default checkpoint directory
    static std::string default_directory();
};
