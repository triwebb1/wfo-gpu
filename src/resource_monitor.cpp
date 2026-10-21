#include "resource_monitor.h"

#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace {

std::mutex              g_limits_mutex;
bool                    g_limits_set = false;
ResourceMonitor::Limits g_limits{0.0f, 100.0f};

float env_float(const char* name, float fallback) {
#ifdef _MSC_VER
    char*  value = nullptr;
    size_t len   = 0;
    if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) return fallback;
    std::string s(value);
    std::free(value);
#else
    const char* value = std::getenv(name);
    if (value == nullptr) return fallback;
    std::string s(value);
#endif
    char*       end = nullptr;
    const float v   = std::strtof(s.c_str(), &end);
    return (end != s.c_str() && std::isfinite(v) && v > 0.0f) ? v : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
ResourceMonitor::Limits ResourceMonitor::default_limits() {
    Limits limits;
    {
        std::lock_guard<std::mutex> lock(g_limits_mutex);
        if (!g_limits_set) {
            const float phys     = physical_ram_gb();
            g_limits.max_ram_gb  = phys > 0.0f ? 0.8f * phys : 16.0f;
            g_limits.max_cpu_pct = 100.0f;
            g_limits_set         = true;
        }
        limits = g_limits;
    }
    limits.max_cpu_pct =
        std::clamp(env_float("WFO_GPU_MAX_CPU_PCT", limits.max_cpu_pct), 1.0f, 100.0f);
    limits.max_ram_gb = env_float("WFO_GPU_MAX_RAM_GB", limits.max_ram_gb);
    return limits;
}

void ResourceMonitor::set_default_limits(const Limits& limits) {
    std::lock_guard<std::mutex> lock(g_limits_mutex);
    g_limits.max_ram_gb  = limits.max_ram_gb;
    g_limits.max_cpu_pct = std::clamp(limits.max_cpu_pct, 1.0f, 100.0f);
    g_limits_set         = true;
}

float ResourceMonitor::physical_ram_gb() {
    constexpr double GB = 1024.0 * 1024.0 * 1024.0;
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return static_cast<float>(ms.ullTotalPhys / GB);
    return 0.0f;
#elif defined(__APPLE__)
    int64_t mem = 0;
    size_t  len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0)
        return static_cast<float>(mem / GB);
    return 0.0f;
#else
    const long pages     = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<float>(static_cast<double>(pages) * page_size / GB);
    return 0.0f;
#endif
}

ResourceMonitor::ResourceMonitor() {
    const Limits limits = default_limits();
    m_max_ram_gb        = limits.max_ram_gb;
    m_max_cpu_pct       = limits.max_cpu_pct;
}

ResourceMonitor::ResourceMonitor(float max_ram_gb, float max_cpu_pct)
    : m_max_ram_gb(max_ram_gb), m_max_cpu_pct(max_cpu_pct) {}

// ---------------------------------------------------------------------------
// current_ram_gb -- process RSS in gigabytes
// ---------------------------------------------------------------------------
float ResourceMonitor::current_ram_gb() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc))) {
        // WorkingSetSize is the current process RSS in bytes
        return static_cast<float>(pmc.WorkingSetSize) / (1024.0f * 1024.0f * 1024.0f);
    }
    return 0.0f;
#elif defined(__APPLE__)
    mach_task_basic_info   info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return static_cast<float>(info.resident_size) / (1024.0f * 1024.0f * 1024.0f);
    }
    return 0.0f;
#else
    // Linux: read VmRSS from /proc/self/status (process RSS in kB)
    std::ifstream status("/proc/self/status");
    if (status.is_open()) {
        std::string line;
        while (std::getline(status, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                // Format: "VmRSS:    <value> kB"
                long kb = 0;
                if (std::sscanf(line.c_str(), "VmRSS: %ld", &kb) == 1 && kb > 0) {
                    return static_cast<float>(kb) / (1024.0f * 1024.0f);  // kB -> GB
                }
            }
        }
    }
    return 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// current_cpu_pct -- approximate CPU utilization
// ---------------------------------------------------------------------------
float ResourceMonitor::current_cpu_pct() const {
#ifdef _WIN32
    // Get process times
    FILETIME creation, exit_time, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit_time, &kernel, &user)) {
        return 0.0f;
    }

    // Get system times
    FILETIME idle, kern, usr;
    if (!GetSystemTimes(&idle, &kern, &usr)) {
        return 0.0f;
    }

    // Convert to uint64_t (100-nanosecond intervals)
    auto to_u64 = [](const FILETIME& ft) -> uint64_t {
        return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };

    uint64_t proc_total = to_u64(kernel) + to_u64(user);
    uint64_t sys_total  = to_u64(kern) + to_u64(usr);

    if (sys_total == 0) return 0.0f;

    // Process CPU as fraction of total system CPU time
    float pct = static_cast<float>(proc_total) / static_cast<float>(sys_total) * 100.0f;
    return std::clamp(pct, 0.0f, 100.0f);
#else
    // Linux: approximate process CPU % from /proc/self/stat and /proc/stat.
    //
    // Read process CPU ticks from /proc/self/stat (fields 14+15 = utime+stime)
    // and total system ticks from /proc/stat (first "cpu" line).
    // This is a snapshot ratio, not a rolling average, mirroring the Windows impl.
    {
        // Read process user+system ticks
        uint64_t proc_utime = 0, proc_stime = 0;
        {
            std::ifstream proc_stat("/proc/self/stat");
            if (proc_stat.is_open()) {
                std::string token;
                // Skip to field 14 (utime) and 15 (stime). Fields are 1-indexed.
                // Field 2 (comm) is in parentheses and may contain spaces, so
                // skip past the closing ')' first.
                std::string content;
                std::getline(proc_stat, content);
                auto paren_end = content.rfind(')');
                if (paren_end != std::string::npos) {
                    // After ')' we have fields 3..N separated by spaces
                    // Fields 14 and 15 correspond to indices 11 and 12 after ')'
                    // (since fields 3..13 are indices 0..10)
                    const char*   p     = content.c_str() + paren_end + 1;
                    unsigned long val   = 0;
                    int           field = 3;  // we're at field 3 now
                    while (*p) {
                        while (*p == ' ') ++p;
                        if (*p == '\0') break;
                        val = std::strtoul(p, const_cast<char**>(&p), 10);
                        if (field == 14) proc_utime = val;
                        if (field == 15) {
                            proc_stime = val;
                            break;
                        }
                        ++field;
                    }
                }
            }
        }

        // Read total system ticks from /proc/stat (first "cpu" line)
        uint64_t sys_total = 0;
        {
            std::ifstream stat_file("/proc/stat");
            if (stat_file.is_open()) {
                std::string line;
                if (std::getline(stat_file, line) && line.compare(0, 4, "cpu ") == 0) {
                    // "cpu  user nice system idle iowait irq softirq steal ..."
                    const char* p = line.c_str() + 4;
                    for (int i = 0; i < 8 && *p; ++i) {
                        while (*p == ' ') ++p;
                        if (*p == '\0') break;
                        unsigned long val = std::strtoul(p, const_cast<char**>(&p), 10);
                        sys_total += val;
                    }
                }
            }
        }

        if (sys_total == 0) return 0.0f;

        uint64_t proc_total = proc_utime + proc_stime;
        float    pct = static_cast<float>(proc_total) / static_cast<float>(sys_total) * 100.0f;
        return std::clamp(pct, 0.0f, 100.0f);
    }
#endif
}

// ---------------------------------------------------------------------------
// within_limits -- check if current usage is within configured limits
// ---------------------------------------------------------------------------
bool ResourceMonitor::within_limits() const {
    if (current_ram_gb() > m_max_ram_gb) return false;
    // CPU check is informational; we enforce via thread count rather than
    // real-time CPU measurement (which is noisy).
    return true;
}

// ---------------------------------------------------------------------------
// max_threads -- worker thread count that honours the CPU limit
// ---------------------------------------------------------------------------
int ResourceMonitor::max_threads() const {
    const int   hw      = hardware_concurrency();
    const float pct     = std::clamp(m_max_cpu_pct, 1.0f, 100.0f);
    const int   threads = static_cast<int>(static_cast<float>(hw) * pct / 100.0f);
    return std::max(threads, 1);
}

// ---------------------------------------------------------------------------
// hardware_concurrency -- number of logical processors
// ---------------------------------------------------------------------------
int ResourceMonitor::hardware_concurrency() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
#else
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return (n > 0) ? n : 1;
#endif
}
