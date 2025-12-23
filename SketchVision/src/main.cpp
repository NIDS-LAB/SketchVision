#include <gflags/gflags.h>

#include "common.hpp"
#include "encoder.hpp"

using namespace std;

DEFINE_string(config, "./configuration/bot/thbot.json",
              "Configuration file location.");

int main(int argc, char *argv[]) {
  __START_FTIMMER__

  google::ParseCommandLineFlags(&argc, &argv, true);

  json config_j;
  try {
    ifstream fin(FLAGS_config, ios::in);
    fin >> config_j;
  } catch (const exception &e) {
    FATAL_ERROR(e.what());
  }

  SketchVision::Encoder sk;
  sk.config_via_json(config_j);
  sk.start();

  __STOP_FTIMER__
  __PRINTF_EXE_TIME__

  return 0;
}
