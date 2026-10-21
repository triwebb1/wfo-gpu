#include "run_lock.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

std::optional<RunLock> RunLock::acquire(const std::string& path, bool force_unlock) {
    std::error_code       ec;
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
#ifdef _WIN32
    // Deleting fails while a live process holds the handle, so this only
    // clears a lock file whose owner is gone.
    if (force_unlock) std::filesystem::remove(p, ec);
#else
    // flock() locks die with their process, so a leftover file is never
    // stale and unlinking it could split two live runs onto different inodes.
    (void)force_unlock;
#endif

    RunLock lock;
    lock.path_ = path;

#ifdef _WIN32
    HANDLE h = ::CreateFileW(p.wstring().c_str(), GENERIC_READ,
                             0,  // no sharing
                             nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
                             nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD err = ::GetLastError();
        std::cerr << "RunLock: cannot lock " << path << " (error " << err << ")";
        if (err == ERROR_FILE_EXISTS || err == ERROR_ALREADY_EXISTS)
            std::cerr << ": another run is active, or a crashed run left the lock";
        std::cerr << std::endl;
        return std::nullopt;
    }
    lock.h_ = h;
#else
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        std::cerr << "RunLock: cannot open " << path << ": " << std::strerror(errno) << std::endl;
        return std::nullopt;
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const int err = errno;
        ::close(fd);
        std::cerr << "RunLock: cannot lock " << path << ": "
                  << (err == EWOULDBLOCK ? "another run is active" : std::strerror(err))
                  << std::endl;
        return std::nullopt;
    }
    // A previous holder may have unlinked the file between our open() and
    // flock(); then we locked an orphaned inode. Only the inode currently at
    // `path` counts.
    struct stat held {
    }, current{};
    if (::fstat(fd, &held) != 0 || ::stat(path.c_str(), &current) != 0 ||
        held.st_ino != current.st_ino || held.st_dev != current.st_dev) {
        ::close(fd);
        std::cerr << "RunLock: " << path << " changed while locking; retry" << std::endl;
        return std::nullopt;
    }
    // Record the owner for anyone inspecting the file.
    const std::string pid = std::to_string(::getpid()) + "\n";
    if (::ftruncate(fd, 0) == 0) (void)!::write(fd, pid.data(), pid.size());
    lock.fd_ = fd;
#endif
    lock.held_ = true;
    return lock;
}

void RunLock::release() noexcept {
    if (!held_) return;
#ifdef _WIN32
    ::CloseHandle(static_cast<HANDLE>(h_));  // FILE_FLAG_DELETE_ON_CLOSE removes the file
    h_ = nullptr;
#else
    // Unlink while still holding the lock so no other process can lock the
    // stale inode we are about to drop.
    ::unlink(path_.c_str());
    ::close(fd_);
    fd_ = -1;
#endif
    held_ = false;
}

RunLock::~RunLock() {
    release();
}

RunLock::RunLock(RunLock&& other) noexcept
    :
#ifdef _WIN32
      h_(std::exchange(other.h_, nullptr)),
#else
      fd_(std::exchange(other.fd_, -1)),
#endif
      held_(std::exchange(other.held_, false)),
      path_(std::move(other.path_)) {
}

RunLock& RunLock::operator=(RunLock&& other) noexcept {
    if (this != &other) {
        release();
#ifdef _WIN32
        h_ = std::exchange(other.h_, nullptr);
#else
        fd_ = std::exchange(other.fd_, -1);
#endif
        held_ = std::exchange(other.held_, false);
        path_ = std::move(other.path_);
    }
    return *this;
}
