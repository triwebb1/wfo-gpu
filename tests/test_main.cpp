// ---------------------------------------------------------------------------
// Test entry point.
//
// Every test process runs inside its own scratch directory: it becomes the
// working directory and the system temp directory, so tests that write
// relative or temp paths never collide when ctest runs them in parallel.
// The early-close calendar is staged next to it, as the build tree does.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

fs::path make_scratch_dir() {
    std::random_device rd;
    const auto         stamp =
        std::chrono::steady_clock::now().time_since_epoch().count() ^ static_cast<long long>(rd());
    const fs::path dir =
        fs::temp_directory_path() / ("wfo_gpu_tests_" + std::to_string(stamp & 0xffffffffffLL));
    fs::create_directories(dir / "config");
    std::error_code ec;
    fs::copy_file(fs::path(WFO_GPU_SOURCE_DIR) / "config" / "cme_early_closes.csv",
                  dir / "config" / "cme_early_closes.csv", fs::copy_options::overwrite_existing,
                  ec);
    return dir;
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    const fs::path original = fs::current_path();
    const fs::path scratch  = make_scratch_dir();
    for (const char* var : {"TMPDIR", "TMP", "TEMP"}) set_env(var, scratch.string());
    fs::current_path(scratch);

    const int rc = RUN_ALL_TESTS();

    fs::current_path(original);
    std::error_code ec;
    fs::remove_all(scratch, ec);
    if (ec) std::cerr << "note: could not remove " << scratch << ": " << ec.message() << "\n";
    return rc;
}
