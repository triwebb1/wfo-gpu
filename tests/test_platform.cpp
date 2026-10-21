// Platform layer: durable atomic writes and the run lock (Windows and POSIX).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "crc32c.h"
#include "durable_file.h"
#include "run_lock.h"

namespace fs = std::filesystem;

namespace {

fs::path scratch(const char* name) {
    const fs::path dir = fs::temp_directory_path() / "wfo_gpu_platform";
    fs::create_directories(dir);
    return dir / name;
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

TEST(DurableFile, WritesAtomicallyAndTracksCrc) {
    const fs::path target = scratch("durable.bin");
    fs::remove(target);
    const std::string payload = "walk-forward checkpoint payload";
    uint32_t          crc     = 0;
    DurableFileWriter::atomic_write(target.string(), [&](DurableFileWriter& w) {
        w.write_bytes(payload.data(), payload.size());
        crc = w.crc32c();
    });
    EXPECT_EQ(slurp(target), payload);
    EXPECT_EQ(crc, crc32c_finalize(crc32c_update(0xFFFFFFFFu, payload.data(), payload.size())));
    EXPECT_FALSE(fs::exists(target.string() + ".tmp"));
}

TEST(DurableFile, ReplacesExistingFile) {
    const fs::path target = scratch("replace.bin");
    DurableFileWriter::atomic_write(target.string(),
                                    [](DurableFileWriter& w) { w.write_bytes("old", 3); });
    DurableFileWriter::atomic_write(target.string(),
                                    [](DurableFileWriter& w) { w.write_bytes("newer", 5); });
    EXPECT_EQ(slurp(target), "newer");
}

TEST(DurableFile, FailedWriteLeavesPreviousVersion) {
    const fs::path target = scratch("rollback.bin");
    DurableFileWriter::atomic_write(target.string(),
                                    [](DurableFileWriter& w) { w.write_bytes("good", 4); });
    EXPECT_THROW(DurableFileWriter::atomic_write(target.string(),
                                                 [](DurableFileWriter& w) {
                                                     w.write_bytes("partial", 7);
                                                     throw std::runtime_error("simulated crash");
                                                 }),
                 std::runtime_error);
    EXPECT_EQ(slurp(target), "good");
    EXPECT_FALSE(fs::exists(target.string() + ".tmp"));
}

TEST(DurableFile, CreatesParentDirectories) {
    const fs::path dir = scratch("nested");
    fs::remove_all(dir);
    const fs::path target = dir / "a" / "b" / "file.bin";
    DurableFileWriter::atomic_write(target.string(),
                                    [](DurableFileWriter& w) { w.write_bytes("x", 1); });
    EXPECT_EQ(slurp(target), "x");
}

TEST(RunLock, IsExclusiveUntilReleased) {
    const fs::path path = scratch("exclusive.lock");
    fs::remove(path);
    {
        auto first = RunLock::acquire(path.string(), false);
        ASSERT_TRUE(first.has_value());
        EXPECT_FALSE(RunLock::acquire(path.string(), false).has_value());

        // Moving the lock keeps it held.
        RunLock moved = std::move(*first);
        first.reset();
        EXPECT_FALSE(RunLock::acquire(path.string(), false).has_value());
    }
    auto again = RunLock::acquire(path.string(), false);
    EXPECT_TRUE(again.has_value());
}

TEST(RunLock, ForceUnlockClearsStaleLockFile) {
    const fs::path path = scratch("stale.lock");
    fs::remove(path);
    { std::ofstream(path) << "12345\n"; }  // left behind by a crashed run
    auto lock = RunLock::acquire(path.string(), true);
    EXPECT_TRUE(lock.has_value());
}
