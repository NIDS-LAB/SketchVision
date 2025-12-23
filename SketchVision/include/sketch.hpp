#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <gflags/gflags_declare.h>
#include <iostream>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <pcap.h>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "flow_define.hpp"
#include "packet_basic.hpp"

#define DET_TH 40

namespace sketch {

struct cord {
  unsigned int x;
  unsigned int y;
};

class sketch_constructor {
private:
  const shared_ptr<vector<shared_ptr<SketchVision::basic_packet>>>
      p_parse_result;
  shared_ptr<vector<shared_ptr<tuple5_flow4>>> p_construct_result4;
  shared_ptr<std::vector<bool>> p_label;

  int image_size;
  std::string attack_name;
  std::string resPath;
  int sketch_memory;
  int K;
  uint32_t crop_size;

  cv::Mat image;
  std::unordered_map<std::string, cv::VideoWriter> flowVid;
  void writeVideo(std::map<std::string, uint32_t> &);

public:
  sketch_constructor(std::string &str, std::string &rs)
      : p_parse_result(nullptr), attack_name(str), resPath(rs) {}
  sketch_constructor(const decltype(p_parse_result) p_parse_result,
                     const decltype(p_label) p_label, std::string &str,
                     std::string &rs, int mem = 0)
      : p_parse_result(p_parse_result), p_label(p_label), attack_name(str),
        resPath(rs), sketch_memory(mem) {

    crop_size = CROP_SIZE;
    K = mem;
    image_size = std::sqrt(sketch_memory * 1024 * 1024 / 3);
    LOGF("2D Sketch size: %d*%d", image_size, image_size);
    image = cv::Mat::zeros(cv::Size(image_size, image_size), CV_8UC4);
  }

  sketch_constructor(const sketch_constructor &) = delete;
  sketch_constructor operator=(const sketch_constructor &) = delete;

  void construct_sketch();
  void construct_flow_pic();
  void save_image(string location);
  void dump_flow_statistic(void) const;
  void get_pic_decode(std::map<std::string, struct cord> &,
                      std::map<std::string, bool> &);
  void get_pic_decode_p(std::map<std::string, struct cord> &,
                        std::map<std::string, bool> &, uint32_t, string &,
                        std::map<std::string, uint32_t> &,
                        std::map<std::string, uint32_t> &);
};

}; // namespace sketch
