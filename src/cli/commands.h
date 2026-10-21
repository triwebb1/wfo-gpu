#pragma once

#include "cli/args.h"

namespace cli {

// Data utilities (data_commands.cpp)
int cmd_generate(Args& args);
int cmd_convert(Args& args);
int cmd_info(Args& args);

// Optimization pipeline (pipeline_commands.cpp)
int cmd_screen(Args& args);
int cmd_walk_forward(Args& args);
int cmd_run(Args& args);
int cmd_reanalyze(Args& args);
int cmd_detail(Args& args);

}  // namespace cli
