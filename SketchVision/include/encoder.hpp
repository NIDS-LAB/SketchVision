#pragma once

#include "basic_dataset.hpp"
#include "common.hpp"
#include "sketch.hpp"

#include <limits.h>
#include <string>
#include <unistd.h>

using namespace std;

namespace SketchVision {

class Encoder {
private:
  json jin_main;
  string file_path = "";
  string out_dir = "../sketch_out/";

  shared_ptr<vector<shared_ptr<basic_packet>>> p_parse_result;
  shared_ptr<binary_label_t> p_label;
  int conf_memory = 0;

public:
  void start(void) {
    __START_FTIMMER__

    if (jin_main["dataset"].count("data_path") &&
        jin_main["dataset"].count("label_path")) {
      LOGF("Load & split datasets.");
      const auto p_dataset_constructor =
          make_shared<basic_dataset>(p_parse_result);
      p_dataset_constructor->configure_via_json(jin_main["dataset"]);
      p_dataset_constructor->import_dataset();
      p_label = p_dataset_constructor->get_label();
      p_parse_result = p_dataset_constructor->get_raw_pkt();
    } else {
      LOGF("Dataset not found.");
    }

    LOGF("#Packets: %ld , #lables: %ld", p_parse_result->size(),
         p_label->size());

    std::string path = jin_main["dataset"]["data_path"];
    std::cout << path << std::endl;
    size_t lastSlash = path.find_last_of("/");
    std::string filename =
        (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
    size_t lastDot = filename.find_last_of(".");
    std::string attack_name =
        (lastDot != std::string::npos) ? filename.substr(0, lastDot) : filename;

    lastSlash = file_path.find_last_of('/');
    std::string fLink = (lastSlash != std::string::npos)
                            ? file_path.substr(lastSlash + 1)
                            : file_path;

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    {
      __START_FTIMMER__
      LOGF("Deriving Clean Flow behavior");
      const auto clean_flows = make_shared<sketch::sketch_constructor>(
          p_parse_result, p_label, attack_name, out_dir);
      clean_flows->construct_flow_pic();

      __STOP_FTIMER__
      __PRINTF_EXE_TIME__
    }

    {
      LOGF("Encoding Flows in SketchVision");
      __START_FTIMMER__
      const auto sketch_constructor = make_shared<sketch::sketch_constructor>(
          p_parse_result, p_label, attack_name, out_dir, conf_memory);
      sketch_constructor->construct_sketch();
      __STOP_FTIMER__
      __PRINTF_EXE_TIME__
    }
  }

  void config_via_json(const json &jin) {
    try {
      if (jin.count("dataset")) {
        jin_main = jin;
      } else {
        throw logic_error("Incomplete json configuration.");
      }
      if (jin.count("memory")) {
        string tmp = static_cast<decltype(tmp)>(jin["memory"]);
        conf_memory = std::stoi(tmp);
      } else {
        conf_memory = 14; // 14 MB default
      }
    } catch (const exception &e) {
      FATAL_ERROR(e.what());
    }
  }
};

} // namespace SketchVision
