#pragma once
// ---------------------------------------------------------------------------
// RunLock -- advisory lock that keeps two processes from resuming into the
// same checkpoint (interleaved writes would silently corrupt resume state).
//
// Windows: CreateFileW with CREATE_NEW + FILE_FLAG_DELETE_ON_CLOSE and no
//   sharing. CREATE_NEW fails if the lock file exists; the OS deletes the
//   file when the handle closes, including on process termination.
// POSIX: open + flock(LOCK_EX | LOCK_NB). The kernel drops the lock when the
//   process exits for any reason, so a crash never leaves a stale lock; the
//   file itself is removed on release.
//
// force_unlock (Windows only) removes a leftover lock file first. Removal
// fails while a live process holds it, so it never breaks an active lock.
// POSIX ignores it: flock() locks cannot outlive their process.
//
// Usage:
//   auto lock = RunLock::acquire("output/checkpoints/screen.lock", false);
//   if (!lock) { error "another run in progress"; return 1; }
//   // ... run ...
//   // lock released in destructor.
// ---------------------------------------------------------------------------

#include <optional>
#include <string>

class RunLock {
public:
    static std::optional<RunLock> acquire(const std::string& path, bool force_unlock);

    ~RunLock();

    // Movable, not copyable.
    RunLock(const RunLock&)            = delete;
    RunLock& operator=(const RunLock&) = delete;
    RunLock(RunLock&& other) noexcept;
    RunLock& operator=(RunLock&& other) noexcept;

private:
    RunLock() = default;
    void release() noexcept;

#ifdef _WIN32
    void* h_ = nullptr;  // HANDLE
#else
    int fd_ = -1;
#endif
    bool        held_ = false;
    std::string path_;
};
