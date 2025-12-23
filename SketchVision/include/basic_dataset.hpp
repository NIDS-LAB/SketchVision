#pragma once

#include "common.hpp"
#include "packet_basic.hpp"

namespace SketchVision {

using binary_label_t = vector<bool>;

class basic_dataset {
private:
  shared_ptr<vector<shared_ptr<basic_packet>>> p_parse_result;
  shared_ptr<binary_label_t> p_label;

  string load_data_path = "";
  string load_label_path = "";

public:
  basic_dataset() {}
  basic_dataset(const decltype(p_parse_result) &p_parse_result)
      : p_parse_result(p_parse_result) {}

  void configure_via_json(const json &jin);

  inline auto get_label(void) const -> decltype(p_label) { return p_label; }

  inline auto get_raw_pkt(void) const -> decltype(p_parse_result) {
    return p_parse_result;
  }

  void import_dataset(void) {
    __START_FTIMMER__
    ifstream _ifd(load_data_path);
    vector<string> string_temp;
    while (true) {
      string _s;
      if (getline(_ifd, _s)) {
        string_temp.push_back(_s);
      } else {
        break;
      }
    }
    _ifd.close();
    size_t num_pkt = string_temp.size();
    p_parse_result =
        make_shared<decltype(p_parse_result)::element_type>(num_pkt);

    const size_t multiplex_num = 64;
    const u_int32_t part_size =
        ceil(((double)num_pkt) / ((double)multiplex_num));
    vector<pair<size_t, size_t>> _assign;
    for (size_t core = 0, idx = 0; core < multiplex_num;
         ++core, idx = min(idx + part_size, num_pkt)) {
      _assign.push_back({idx, min(idx + part_size, num_pkt)});
    }
    mutex add_m;
    auto __f = [&](size_t _from, size_t _to) -> void {
      for (size_t i = _from; i < _to; ++i) {
        const string &str = string_temp[i];
        if (str[0] == '4') {
          const auto make_pkt = make_shared<basic_packet4>(str);
          p_parse_result->at(i) = make_pkt;
        } else {
          const auto make_pkt = make_shared<basic_packet_bad>();
          p_parse_result->at(i) = make_pkt;
        }
      }
    };

    vector<thread> vt;
    for (size_t core = 0; core < multiplex_num; ++core) {
      vt.emplace_back(__f, _assign[core].first, _assign[core].second);
    }

    for (auto &t : vt)
      t.join();

    ifstream _ifl(load_label_path);
    p_label = make_shared<decltype(p_label)::element_type>();
    string ll;
    _ifl >> ll;
    for (const char a : ll) {
      p_label->push_back(a == '1');
    }
    _ifl.close();
    assert(p_label->size() == p_parse_result->size());

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__
  }
};

} // namespace SketchVision
