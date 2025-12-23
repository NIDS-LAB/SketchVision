#include "sketch.hpp"
#include "hash.hpp"
#include <boost/mpl/assert.hpp>
#include <boost/range/detail/sfinae.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace sketch;
using namespace SketchVision;

inline auto reverse_bits(uint8_t n) -> uint8_t {
  uint8_t result = 0;
  for (int i = 0; i < 6; ++i) {
    result <<= 1;
    result |= (n & 1);
    n >>= 1;
  }
  return result;
};

bool MKDIR(const string &path) {

  try {
    // create_directories creates all missing parents
    if (std::filesystem::create_directories(path)) {
      LOGF("Created directory: %s", path.data());
    }
    return true; // exists now or already existed
  } catch (const std::filesystem::filesystem_error &e) {
    LOGF("Failed to create directory: %s. \n Reason: %s ", path.data(),
         e.what());
    return false;
  }
}

void sketch_constructor::writeVideo(std::map<std::string, uint32_t> &flCnt) {
  for (auto fl : flowVid) {
    if (flCnt[fl.first] >= DET_TH)
      fl.second.release();
  }
  flowVid.clear();
}

template <class T>
void dump_map(std::string path, std::map<std::string, T> &map_) {

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    LOGF("Error opening the file: %s - %s", path.data(), std::strerror(errno));
  }

  size_t size = map_.size();
  f.write(reinterpret_cast<const char *>(&size), sizeof(size));

  for (const auto &pair : map_) {
    size_t strsize = pair.first.size();
    f.write(reinterpret_cast<const char *>(&strsize), sizeof(strsize));
    f.write(pair.first.data(), strsize);

    f.write(reinterpret_cast<const char *>(&pair.second), sizeof(pair.second));
  }
  f.close();
}
void load_map(std::string path, std::map<std::string, uint32_t> &map_) {

  std::ifstream f(path, std::ios::binary);

  size_t size;
  f.read(reinterpret_cast<char *>(&size), sizeof(size));

  for (size_t i = 0; i < size; ++i) {
    size_t strSize;
    f.read(reinterpret_cast<char *>(&strSize),
           sizeof(strSize)); // Read value size
    std::string key(strSize, '\0');
    f.read(&key[0], strSize); // Read value
    uint32_t value;
    f.read(reinterpret_cast<char *>(&value), sizeof(value)); // Read key

    map_[key] = value;
  }

  f.close();
}

void sketch_constructor::construct_flow_pic() {

  std::map<std::string, uint64_t> flows;
  std::map<std::string, uint64_t> flows_last;
  std::map<std::string, bool> flow_ben;

  std::map<std::string, uint32_t> flow_cnt;
  std::map<std::string, std::vector<uint64_t>> flow_time;
  std::map<std::string, std::vector<uint16_t>> flow_pkts;

  std::map<std::string, uint32_t> flowidBenIdx;
  std::map<std::string, uint32_t> flowidAtkIdx;

  for (uint32_t i = 0; i < p_parse_result->size(); i++) {
    auto p_rep = p_parse_result->at(i);
    if (typeid(*p_rep) == typeid(basic_packet_bad)) {
      continue;
    }
    if (typeid(*p_rep) == typeid(basic_packet4)) {
      const auto _p_rep = dynamic_pointer_cast<basic_packet4>(p_rep);

      uint32_t src_ip = tuple_get_src_addr(_p_rep->flow_id);
      uint32_t dst_ip = tuple_get_dst_addr(_p_rep->flow_id);
      uint16_t src_port = tuple_get_src_port(_p_rep->flow_id);
      uint16_t dst_port = tuple_get_dst_port(_p_rep->flow_id);
      uint16_t len_ = _p_rep->len;
      uint16_t q_len = static_cast<uint16_t>((len_) / 1);

      if (len_ > 1503) // Assume MTU 1500
        continue;

      std::string id = std::to_string(src_ip) + std::to_string(dst_ip);

      const uint64_t _timestp =
          ((_p_rep->ts.tv_nsec * 1e-3) + _p_rep->ts.tv_sec * 1e6);

      if (flows.find(id) == flows.end()) {
        flows[id] = _timestp;
        flow_ben[id] = p_label->at(i);
      }

      uint64_t d_timestp = (_timestp - flows[id]);

      flow_cnt[id]++;
      flow_pkts[id].push_back(q_len);
      flow_time[id].push_back(d_timestp);
    }
  }

  uint32_t ben = 1;
  uint32_t atk = 1;
  std::string path = resPath + "/Clean_Stream/" + attack_name;

  MKDIR(path);

  for (auto &pair : flow_time) {
    if (pair.second.size() < DET_TH) { // Mice flows are not intrested
      continue;
    }

    cv::Mat image_ = cv::Mat::zeros(cv::Size(crop_size, crop_size), CV_8UC4);
    uint32_t im_size = crop_size;

    uint32_t CNT = 0;
    uint64_t sum = 0;
    for (auto &ts : pair.second) {

      CNT++;
      sum = ts;

      double angle = std::sqrt(ts / 1000);
      double radius = std::sqrt(sum / 2000 / CNT);

      unsigned int p_x =
          static_cast<unsigned int>((double)im_size / 2 + radius * cos(angle)) %
          image_.cols;
      unsigned int p_y =
          static_cast<unsigned int>((double)im_size / 2 + radius * sin(angle)) %
          image_.rows;

      p_x = (p_x > 0) ? p_x : ((p_x + image_.cols) % image_.cols);
      p_y = (p_y > 0) ? p_y : ((p_y + image_.rows) % image_.rows);

      auto val1 = flow_pkts[pair.first][CNT - 1];
      auto &pix = image_.at<cv::Vec4b>(p_x, p_y);
      uint16_t col_pix = (reverse_bits(pix[1] & 0x3F) * 256) + pix[2];
      if (col_pix == 0) {
        pix[0] = val1 & 0xFF;
        pix[1] = ((val1 >> 8) & 0x03) << 6;
        pix[2] = 8;
        pix[3] = 255;
      } else if (col_pix < 2048 - 8) {
        uint16_t b_pkts = (((pix[1] & 0xC0) >> 6) << 8) | pix[0];
        b_pkts = (b_pkts * (col_pix / 8) + val1) / (col_pix / 8 + 1);
        pix[0] = b_pkts & 0xFF;
        col_pix += 8;
        pix[2] = col_pix & 0xFF;
        pix[1] =
            ((b_pkts >> 8) & 0x03) << 6 | reverse_bits((col_pix >> 8) & 0x3F);
      }
      std::string fname = "";
      if (flow_ben[pair.first] == false) {
        flowidBenIdx[pair.first] = ben;
        fname += "/Ben_" + std::to_string(ben);
      }
      if (flow_ben[pair.first] == true) {
        flowidAtkIdx[pair.first] = atk;
        fname += "/Atk_" + std::to_string(atk);
      }
      auto p = path + fname;
      if (!flowVid.count(pair.first))
        flowVid[pair.first] = cv::VideoWriter(
            p + ".mkv", cv::VideoWriter::fourcc('F', 'F', 'V', '1'), 30,
            cv::Size(crop_size, crop_size));

      // auto &vid = flowVid[pair.first];
      // if (!vid.isOpened()) {
      //   LOGF("Failed to open video: %s.mkv ", p.data());
      // }

      cv::Mat image_bgr;
      cv::cvtColor(image_, image_bgr, cv::COLOR_BGRA2BGR);
      flowVid[pair.first].write(image_bgr);
    }

    std::string fname = "";
    if (flow_ben[pair.first] == false) {
      flowidBenIdx[pair.first] = ben;
      fname += "/Ben_" + std::to_string(ben);
      ben++;
    }
    if (flow_ben[pair.first] == true) {
      flowidAtkIdx[pair.first] = atk;
      fname += "/Atk_" + std::to_string(atk);
      atk++;
    }
    // save the last image
    // auto p = path + fname + ".png";
    // cv::imwrite(p, image_);
  }
  dump_map(path + "/ben.bin", flowidBenIdx);
  dump_map(path + "/atk.bin", flowidAtkIdx);
  dump_map(path + "/flCnt.bin", flow_cnt);
  writeVideo(flow_cnt);
}

void sketch_constructor::construct_sketch() {

  std::map<std::string, struct cord> flow_cord;
  std::map<std::string, uint64_t> flows;
  std::map<std::string, uint32_t> flow_cnt;
  std::map<std::string, bool> flow_ben;

  std::string path_ = resPath + "/Clean_Stream/" + attack_name;

  std::map<std::string, uint32_t> flowidBenIdx;
  std::map<std::string, uint32_t> flowidAtkIdx;
  std::map<std::string, uint32_t> flowidCntGr;

  load_map(path_ + "/ben.bin", flowidBenIdx);
  load_map(path_ + "/atk.bin", flowidAtkIdx);
  load_map(path_ + "/flCnt.bin", flowidCntGr);
  uint32_t im_size = crop_size;

  for (uint32_t i = 0; i < p_parse_result->size(); i++) {
    auto p_rep = p_parse_result->at(i);
    if (typeid(*p_rep) == typeid(basic_packet_bad)) {
      continue;
    }
    if (typeid(*p_rep) == typeid(basic_packet4)) {
      const auto _p_rep = dynamic_pointer_cast<basic_packet4>(p_rep);

      uint32_t src_ip = tuple_get_src_addr(_p_rep->flow_id);
      uint32_t dst_ip = tuple_get_dst_addr(_p_rep->flow_id);
      uint16_t src_port = tuple_get_src_port(_p_rep->flow_id);
      uint16_t dst_port = tuple_get_dst_port(_p_rep->flow_id);
      uint16_t len_ = _p_rep->len;
      uint16_t q_len = static_cast<uint16_t>((len_) / 1);

      if (len_ > 1503)
        continue;

      std::string id = std::to_string(src_ip) + std::to_string(dst_ip);
      uint64_t _timestp = ((_p_rep->ts.tv_nsec * 1e-9) + _p_rep->ts.tv_sec) *
                          1e6; // GET_DOUBLE_TS(_p_rep->ts);

      if (flows.find(id) == flows.end()) {
        flows[id] = _timestp;
        flow_cnt[id] = 0;
        flow_ben[id] = p_label->at(i);
      }

      flow_cnt[id]++;

      unsigned int x =
          hash_fn_xor(src_ip, dst_ip, 17, 9, 3, 937529) % image.cols;
      unsigned int y =
          hash_fn_xor(src_ip, dst_ip, 17, 7, 23, 359873) % image.rows;

      auto d_timestp = (_timestp - flows[id]);

      double angle = std::sqrt(d_timestp / 1000);
      double radius = std::sqrt(d_timestp / 2000 / flow_cnt[id]);

      int x_ = static_cast<int>(x + radius * cos(angle)) % image.cols;
      int y_ = static_cast<int>(y + radius * sin(angle)) % image.rows;

      unsigned int p_x =
          static_cast<unsigned int>((double)im_size / 2 + radius * cos(angle)) %
          im_size;
      unsigned int p_y =
          static_cast<unsigned int>((double)im_size / 2 + radius * sin(angle)) %
          im_size;
      p_x = (x - im_size / 2 + p_x) % image.cols;
      p_y = (y - im_size / 2 + p_y) % image.rows;

      uint16_t val1 = q_len;
      auto &pix = image.at<cv::Vec4b>(p_x, p_y);
      uint16_t col_pix = (reverse_bits(pix[1] & 0x3F) * 256) + pix[2];
      if (col_pix == 0) {
        pix[0] = val1 & 0xFF;
        pix[1] = ((val1 >> 8) & 0x03) << 6;
        pix[2] = 8;
        pix[3] = 255;
      } else if (col_pix < 2048 - 8) {
        uint16_t b_pkts = (((pix[1] & 0xC0) >> 6) << 8) | pix[0];
        b_pkts = (b_pkts * (col_pix / 8) + val1) / (col_pix / 8 + 1);
        pix[0] = b_pkts & 0xFF; // & 0xFF;
        col_pix += 8;
        pix[2] = col_pix & 0xFF;
        pix[1] =
            ((b_pkts >> 8) & 0x03) << 6 | reverse_bits((col_pix >> 8) & 0x3F);
      }
      if (flowidCntGr[id] >= DET_TH) {
        flow_cord[id] = {x, y};
        get_pic_decode_p(flow_cord, flow_ben, flow_cnt[id], id, flowidBenIdx,
                         flowidAtkIdx);
      }
    }
  }
  writeVideo(flow_cnt);
  save_image(resPath + "/" + attack_name + "_" + std::to_string(K) + ".png");
}

void sketch_constructor::get_pic_decode_p(
    std::map<std::string, struct cord> &flow_cord,
    std::map<std::string, bool> &flow_label, uint32_t nPacket, string &id,
    std::map<std::string, uint32_t> &flowidBenIdx,
    std::map<std::string, uint32_t> &flowidAtkIdx) {

  std::string path =
      resPath + "/Sketch_Stream_" + std::to_string(K) + "/" + attack_name;

  MKDIR(path);

  int small_img_size = crop_size;
  cv::Mat image_ =
      cv::Mat::zeros(cv::Size(small_img_size, small_img_size), CV_8UC4);
  auto center_x = flow_cord[id].x;
  auto center_y = flow_cord[id].y;
  for (int i = (-1 * small_img_size / 2); i < (small_img_size / 2); i++) {
    for (int j = (-1 * small_img_size / 2); j < (small_img_size / 2); j++) {

      auto r = image.at<cv::Vec4b>((center_x + i) % image_size,
                                   (center_y + j) % image_size)[0];
      auto g = image.at<cv::Vec4b>((center_x + i) % image_size,
                                   (center_y + j) % image_size)[1];
      auto b = image.at<cv::Vec4b>((center_x + i) % image_size,
                                   (center_y + j) % image_size)[2];

      image_.at<cv::Vec4b>(i + small_img_size / 2, j + small_img_size / 2) =
          cv::Vec4b(r, g, b, 255);
    }
  }

  std::string fname = "";
  if (flow_label[id] == false) {
    if (flowidBenIdx.find(id) == flowidBenIdx.end()) {
      LOGF("Didn't find id ben %s", id.data());
      return;
    }

    fname += "/Ben_" + std::to_string(flowidBenIdx[id]);
  }
  if (flow_label[id] == true) {
    if (flowidAtkIdx.find(id) == flowidAtkIdx.end()) {
      LOGF("Didn't find id atk %s", id.data());
      return;
    }
    fname += "/Atk_" + std::to_string(flowidAtkIdx[id]);
  }

  auto p = path + fname;
  if (!flowVid.count(id))
    flowVid[id] =
        cv::VideoWriter(p + ".mkv", cv::VideoWriter::fourcc('F', 'F', 'V', '1'),
                        30, cv::Size(crop_size, crop_size));
  cv::Mat image_bgr;
  cv::cvtColor(image_, image_bgr, cv::COLOR_BGRA2BGR);
  flowVid[id].write(image_bgr);
  // cv::imwrite(p, image_);
}

void sketch_constructor::save_image(std::string location) {
  if (!image.empty()) {
    LOGF("Sketch Image Saved in location ");
    cv::imwrite(location, image);
  } else {
    LOGF("Empty Image");
  }
}
