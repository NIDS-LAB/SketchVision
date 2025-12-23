#include "basic_dataset.hpp"

using namespace SketchVision;

void basic_dataset::configure_via_json(const json &jin) {
  try {

    if (jin.count("data_path")) {
      load_data_path = static_cast<decltype(load_data_path)>(jin["data_path"]);
    }
    if (jin.count("label_path")) {
      load_label_path =
          static_cast<decltype(load_label_path)>(jin["label_path"]);
    }

  } catch (const exception &e) {
    FATAL_ERROR(e.what());
  }
}
