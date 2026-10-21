#pragma once
// ---------------------------------------------------------------------------
// ResourceMonitor -- RAM and CPU limits for long-running runs
//
// Worker pools size themselves from max_threads(), so the CPU limit is
// enforced through thread count. RAM is checked against process RSS.
//
// Process-wide defaults come from set_default_limits() (the CLI applies the
// [resources] config section) and can be overridden with the environment
// variables WFO_GPU_MAX_CPU_PCT and WFO_GPU_MAX_RAM_GB.
// ---------------------------------------------------------------------------

#include <cstdint>

class ResourceMonitor {
public:
    struct Limits {
        float max_ram_gb;   // maximum process RSS in gigabytes
        float max_cpu_pct;  // share of logical processors to use, 1-100
    };

    // Uses the process-wide default limits.
    ResourceMonitor();
    ResourceMonitor(float max_ram_gb, float max_cpu_pct);

    // Process-wide defaults. Initially 100% CPU and 80% of physical memory,
    // then environment overrides; set_default_limits() replaces both values
    // but environment overrides still win.
    static Limits default_limits();
    static void   set_default_limits(const Limits& limits);

    // Total physical memory in gigabytes (0 if unknown).
    static float physical_ram_gb();

    // Current process RSS in gigabytes.
    float current_ram_gb() const;

    // Process share of total CPU time since start (0-100). A snapshot, not
    // a rolling average.
    float current_cpu_pct() const;

    // Check if current usage is within configured limits.
    bool within_limits() const;

    // Recommended maximum number of worker threads:
    // hardware_concurrency() * max_cpu_pct / 100, at least 1.
    int max_threads() const;

    // Configured limits (read-only accessors)
    float max_ram_gb() const { return m_max_ram_gb; }
    float max_cpu_pct() const { return m_max_cpu_pct; }

    // Number of logical processors on the system
    static int hardware_concurrency();

private:
    float m_max_ram_gb;
    float m_max_cpu_pct;
};
