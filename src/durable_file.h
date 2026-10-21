#pragma once
// ---------------------------------------------------------------------------
// DurableFileWriter -- crash-safe atomic file writer with streaming CRC.
//
//   1. Write to a .tmp file next to the target path.
//   2. Flush the data to stable storage (FlushFileBuffers on Windows,
//      fsync / F_FULLFSYNC on POSIX).
//   3. Atomically replace the final path (MoveFileExW with WRITE_THROUGH on
//      Windows; rename() followed by an fsync of the directory on POSIX).
//   4. Streaming CRC32C over every byte written; callers can stamp the
//      computed CRC into the file before commit.
//
// A crash at any point leaves either the old file or the new one, never a
// torn mix. RAII: if the writer is destroyed without commit() (normal flow
// or exception), the destructor closes the handle and unlinks the .tmp file.
//
// Usage (throws std::runtime_error on I/O failure):
//
//   DurableFileWriter::atomic_write(final_path, [&](DurableFileWriter& w) {
//       w.write_bytes(buf, len);
//       // optionally: uint32_t crc = w.crc32c();
//       // w.write_bytes(&crc, sizeof(crc));
//   });
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

class DurableFileWriter {
public:
    // Run `fn` inside a durable write transaction. On throw, the .tmp file
    // is unlinked and the final path is left untouched.
    static void atomic_write(const std::string&                             final_path,
                             const std::function<void(DurableFileWriter&)>& fn);

    // Stream bytes to the .tmp file and update running CRC32C.
    void write_bytes(const void* data, std::size_t len);

    // Current CRC32C of all bytes written so far (finalized value).
    uint32_t crc32c() const;

    // Current running (non-finalized) CRC state -- used internally.
    uint32_t crc_state() const { return crc_state_; }

    // Non-copyable, non-movable; lifetime is bounded by atomic_write's lambda.
    DurableFileWriter(const DurableFileWriter&)            = delete;
    DurableFileWriter& operator=(const DurableFileWriter&) = delete;
    DurableFileWriter(DurableFileWriter&&)                 = delete;
    DurableFileWriter& operator=(DurableFileWriter&&)      = delete;

    ~DurableFileWriter();

private:
    explicit DurableFileWriter(const std::string& final_path);
    void commit();
    void close_handle() noexcept;

    // Windows HANDLE (as void*) or POSIX file descriptor, kept opaque so
    // this header does not pull in platform headers.
#ifdef _WIN32
    void* h_ = nullptr;
#else
    int fd_ = -1;
#endif
    bool        open_ = false;
    std::string tmp_path_;
    std::string final_path_;
    uint32_t    crc_state_ = 0xFFFFFFFFu;
    bool        committed_ = false;
};
