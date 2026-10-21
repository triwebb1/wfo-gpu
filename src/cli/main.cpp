// wfo-gpu command-line entry point.

#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "cli/commands.h"

#ifndef WFO_GPU_VERSION
#define WFO_GPU_VERSION "0.0.0"
#endif

namespace {

constexpr const char* USAGE =
    R"(wfo-gpu: GPU-accelerated walk-forward optimization for trading strategies

Usage: wfo-gpu <command> [options]

Data
  generate      Write a synthetic bar file
                  --out FILE [--days N] [--seed N] [--start YYYY-MM-DD]
                  [--calendar us_futures|continuous] [--timeframe SEC]
                  [--price P] [--tick T] [--vol V] [--drift D]
                  [--trend K] [--trend-bars N] [--symbol S]
  convert       Convert CSV bars (timestamp,open,high,low,close,volume)
                  --csv FILE --out FILE [--symbol S] [--timeframe SEC]
  info          Show compute devices and, optionally, a bar file summary
                  [--bars FILE] [--session us_futures|continuous]
                  [--signals]       list the signal catalogue strategies draw from
                  [--strategy ID]   decode a strategy id into its rules

Optimization (all take --config FILE)
  screen        Sweep the full strategy grid over the screen range
  walk-forward  Re-optimize the screened set on rolling in-sample windows,
                then trade each winner on the following out-of-sample window
  run           screen, then walk-forward
  reanalyze     Re-apply floors to a finished screen  [--robustness-floor X]
  detail        Full trade logs for chosen strategies
                  --ids FILE [--start BAR] [--end BAR] [--lag 1|2]

Common options for optimization commands
  --bars FILE           override [data] bars
  --output DIR          override [data] output
  --device NAME         auto | gpu | opencl-cpu | native
  --max-strategies N    cap on strategy combinations in the screen
  --resume              continue from the last checkpoint
  --fresh               discard checkpoints and start over
  --force-unlock        remove a stale run lock left by a crashed run

Examples
  wfo-gpu generate --out data/synthetic.bin --days 260
  wfo-gpu run --config config/example.toml
  wfo-gpu info --bars data/synthetic.bin
)";

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0 ||
        std::strcmp(argv[1], "help") == 0) {
        std::cout << USAGE;
        return 0;
    }
    if (std::strcmp(argv[1], "--version") == 0) {
        std::cout << "wfo-gpu " << WFO_GPU_VERSION
#ifdef HAS_OPENCL
                  << " (OpenCL backend built in; needs an OpenCL runtime to use a device)\n";
#else
                  << " (CPU only: built without OpenCL)\n";
#endif
        return 0;
    }

    const std::string command = argv[1];
    cli::Args         args(argc, argv, 2);
    try {
        if (command == "generate") return cli::cmd_generate(args);
        if (command == "convert") return cli::cmd_convert(args);
        if (command == "info") return cli::cmd_info(args);
        if (command == "screen") return cli::cmd_screen(args);
        if (command == "walk-forward") return cli::cmd_walk_forward(args);
        if (command == "run") return cli::cmd_run(args);
        if (command == "reanalyze") return cli::cmd_reanalyze(args);
        if (command == "detail") return cli::cmd_detail(args);
        std::cerr << "wfo-gpu: unknown command '" << command << "'. See wfo-gpu --help.\n";
        return 2;
    } catch (const cli::UsageError& e) {
        std::cerr << "wfo-gpu " << command << ": " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "wfo-gpu " << command << ": " << e.what() << "\n";
        return 1;
    }
}
