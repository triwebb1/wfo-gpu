#include "durable_file.h"
#include "crc32c.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

[[noreturn]] void throw_io(const char* what, long code, const std::string& path) {
    throw std::runtime_error(std::string("DurableFileWriter: ") + what + " failed (error " +
                             std::to_string(code) + ") for " + path);
}

#ifdef _WIN32
std::wstring to_wide(const std::string& s) {
    return std::filesystem::path(s).wstring();
}
#else
// Flush file data and metadata to stable storage. On macOS fsync() only
// reaches the drive cache; F_FULLFSYNC asks the drive to flush as well.
int full_sync(int fd) {
#ifdef F_FULLFSYNC
    if (::fcntl(fd, F_FULLFSYNC) == 0) return 0;
#endif
    return ::fsync(fd);
}

// Persist a rename by syncing the directory that holds the entry.
void sync_parent_dir(const std::string& path) {
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (dir.empty()) dir = ".";
    const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;  // best effort: some filesystems refuse directory opens
    (void)::fsync(fd);
    ::close(fd);
}
#endif

}  // namespace

DurableFileWriter::DurableFileWriter(const std::string& final_path)
    : tmp_path_(final_path + ".tmp"), final_path_(final_path) {
    // Ensure the parent directory exists (best effort).
    std::error_code       ec;
    std::filesystem::path p(final_path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);

#ifdef _WIN32
    HANDLE h = ::CreateFileW(to_wide(tmp_path_).c_str(), GENERIC_WRITE,
                             0,  // no sharing: serializes writers on the same tmp
                             nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw_io("CreateFileW", ::GetLastError(), tmp_path_);
    h_ = h;
#else
    fd_ = ::open(tmp_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd_ < 0) throw_io("open", errno, tmp_path_);
#endif
    open_ = true;
}

void DurableFileWriter::close_handle() noexcept {
    if (!open_) return;
#ifdef _WIN32
    ::CloseHandle(static_cast<HANDLE>(h_));
    h_ = nullptr;
#else
    ::close(fd_);
    fd_ = -1;
#endif
    open_ = false;
}

DurableFileWriter::~DurableFileWriter() {
    close_handle();
    if (!committed_) {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::path(tmp_path_), ec);  // best effort
    }
}

void DurableFileWriter::write_bytes(const void* data, std::size_t len) {
    if (!open_) throw std::runtime_error("DurableFileWriter::write_bytes after close");
    const char* p         = static_cast<const char*>(data);
    std::size_t remaining = len;
    while (remaining > 0) {
        // Chunk large writes: WriteFile takes a 32-bit length.
        const std::size_t chunk = remaining > 0x40000000u ? 0x40000000u : remaining;
#ifdef _WIN32
        DWORD      written = 0;
        const BOOL ok =
            ::WriteFile(static_cast<HANDLE>(h_), p, static_cast<DWORD>(chunk), &written, nullptr);
        if (!ok || written != chunk) throw_io("WriteFile", ::GetLastError(), tmp_path_);
        const std::size_t done = written;
#else
        const ssize_t written = ::write(fd_, p, chunk);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw_io("write", errno, tmp_path_);
        }
        const std::size_t done = static_cast<std::size_t>(written);
#endif
        crc_state_ = crc32c_update(crc_state_, p, done);
        p += done;
        remaining -= done;
    }
}

uint32_t DurableFileWriter::crc32c() const {
    return crc32c_finalize(crc_state_);
}

void DurableFileWriter::commit() {
    if (committed_) return;
    if (!open_) throw std::runtime_error("DurableFileWriter::commit on closed handle");

#ifdef _WIN32
    // Force data through the OS cache to disk before the rename.
    if (!::FlushFileBuffers(static_cast<HANDLE>(h_)))
        throw_io("FlushFileBuffers", ::GetLastError(), tmp_path_);
    if (!::CloseHandle(static_cast<HANDLE>(h_))) {
        open_ = false;
        h_    = nullptr;
        throw_io("CloseHandle", ::GetLastError(), tmp_path_);
    }
    open_ = false;
    h_    = nullptr;
    // MOVEFILE_WRITE_THROUGH returns only after the rename is on disk.
    if (!::MoveFileExW(to_wide(tmp_path_).c_str(), to_wide(final_path_).c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw_io("MoveFileExW", ::GetLastError(), final_path_);
#else
    if (full_sync(fd_) != 0) throw_io("fsync", errno, tmp_path_);
    const int close_rc = ::close(fd_);
    open_              = false;
    fd_                = -1;
    if (close_rc != 0) throw_io("close", errno, tmp_path_);
    // rename() atomically replaces the target; syncing the directory makes
    // the new entry survive a power loss.
    if (::rename(tmp_path_.c_str(), final_path_.c_str()) != 0)
        throw_io("rename", errno, final_path_);
    sync_parent_dir(final_path_);
#endif
    committed_ = true;
}

void DurableFileWriter::atomic_write(const std::string&                             final_path,
                                     const std::function<void(DurableFileWriter&)>& fn) {
    DurableFileWriter w(final_path);
    fn(w);       // may throw -> destructor deletes tmp
    w.commit();  // may throw -> destructor deletes tmp if still present
}
