#pragma once

#include "common.hpp"
#include <netinet/in.h>
#include <pcapplusplus/IPv4Layer.h>
#include <pcapplusplus/IPv6Layer.h>
#include <pcapplusplus/Packet.h>
#include <pcapplusplus/PcapFileDevice.h>
#include <pcapplusplus/TcpLayer.h>
#include <pcapplusplus/UdpLayer.h>

using namespace std;

namespace SketchVision {

using pkt_addr4_t = u_int32_t;
using pkt_len_t = u_int16_t;
using pkt_port_t = u_int16_t;
using pkt_ts_t = timespec;

using pkt_code_t = u_int16_t;
enum pkt_type_t : u_int8_t {
  IPv4,
  IPv6,
  ICMP,
  IGMP,
  TCP_SYN,
  TCP_ACK,
  TCP_FIN,
  TCP_RST,
  UDP,
  UNKNOWN
};
const vector<const char *> type2name = {
    "IPv4",    "IPv6",    "ICMP",    "IGMP", "TCP_SYN",
    "TCP_ACK", "TCP_FIN", "TCP_RST", "UDP",  "UNKNOWN"};
inline void set_pkt_type_code(pkt_code_t &cd, const pkt_type_t t) {
  cd |= (1 << t);
}
inline auto get_pkt_type_code(const pkt_type_t t) -> pkt_code_t {
  return (1 << t);
}
inline auto test_pkt_type_code(const pkt_code_t cd, const pkt_type_t t)
    -> bool {
  return cd & (1 << t);
}

using stack_code_t = u_int16_t;
enum stack_type_t : u_int8_t {
  F_ICMP,
  F_IGMP,
  F_TCP,
  F_UDP,
  F_UNKNOWN,
};
const vector<const char *> stack2name = {
    "ICMP", "IGMP", "TCP", "UDP", "UNKNOWN",
};
inline auto get_pkt_stack_code(const stack_type_t st) -> stack_code_t {
  return (1 << st);
}
inline auto convert_packet2stack_code(const pkt_code_t pc) -> stack_code_t {
  if (test_pkt_type_code(pc, pkt_type_t::ICMP)) {
    return get_pkt_stack_code(stack_type_t::F_ICMP);
  }
  if (test_pkt_type_code(pc, pkt_type_t::IGMP)) {
    return get_pkt_stack_code(stack_type_t::F_IGMP);
  }
  if (test_pkt_type_code(pc, pkt_type_t::UDP)) {
    return get_pkt_stack_code(stack_type_t::F_UDP);
  }
  if (test_pkt_type_code(pc, pkt_type_t::UNKNOWN)) {
    return get_pkt_stack_code(stack_type_t::F_UNKNOWN);
  }
  // assert(test_pkt_type_code(pc, pkt_type_t::TCP_ACK) ||
  // test_pkt_type_code(pc, pkt_type_t::TCP_SYN) ||
  //         test_pkt_type_code(pc, pkt_type_t::TCP_FIN) ||
  //         test_pkt_type_code(pc, pkt_type_t::TCP_RST));
  return get_pkt_stack_code(stack_type_t::F_TCP);
}

using tuple2_conn4 = tuple<pkt_addr4_t, pkt_addr4_t>;
using tuple4_conn4 = tuple<pkt_addr4_t, pkt_addr4_t, pkt_port_t, pkt_port_t>;
using tuple5_conn4 =
    tuple<pkt_addr4_t, pkt_addr4_t, pkt_port_t, pkt_port_t, stack_code_t>;

inline auto tuple_get_src_addr(const tuple2_conn4 &cn) -> pkt_addr4_t {
  return get<0>(cn);
}
inline auto tuple_get_src_addr(const tuple4_conn4 &cn) -> pkt_addr4_t {
  return get<0>(cn);
}
inline auto tuple_get_src_addr(const tuple5_conn4 &cn) -> pkt_addr4_t {
  return get<0>(cn);
}
inline auto get_str_addr(const pkt_addr4_t ad) -> string {
  return pcpp::IPv4Address(ad).toString();
}

inline auto tuple_conn_reverse(const tuple5_conn4 &cn) -> tuple5_conn4 {
  return make_tuple(get<1>(cn), get<0>(cn), get<3>(cn), get<2>(cn), get<4>(cn));
}

inline auto convert_str_addr4(const string &str) -> pkt_addr4_t {
  pcpp::IPv4Address pcpp_ip(str);
  if (str == "0.0.0.0") {
    return 0;
  }
  if (!pcpp_ip.isValid()) {
    FATAL_ERROR("Invalid IPv4");
  } else {
    return pcpp_ip.toInt();
  }
}

inline auto tuple_get_dst_addr(const tuple2_conn4 &cn) -> pkt_addr4_t {
  return get<1>(cn);
}
inline auto tuple_get_dst_addr(const tuple4_conn4 &cn) -> pkt_addr4_t {
  return get<1>(cn);
}
inline auto tuple_get_dst_addr(const tuple5_conn4 &cn) -> pkt_addr4_t {
  return get<1>(cn);
}

inline auto tuple_get_src_port(const tuple4_conn4 &cn) -> pkt_port_t {
  return get<2>(cn);
}
inline auto tuple_get_src_port(const tuple5_conn4 &cn) -> pkt_port_t {
  return get<2>(cn);
}
inline auto tuple_get_dst_port(const tuple4_conn4 &cn) -> pkt_port_t {
  return get<3>(cn);
}
inline auto tuple_get_dst_port(const tuple5_conn4 &cn) -> pkt_port_t {
  return get<3>(cn);
}

inline auto tuple_get_stack(const tuple5_conn4 &cn) -> stack_code_t {
  return get<4>(cn);
}

inline auto tuple_is_stack(const tuple5_conn4 &cn, const stack_type_t tp)
    -> bool {
  return get<4>(cn) & (1 << tp);
}

inline auto tuple4_extend(const tuple4_conn4 &cn, const stack_code_t sc)
    -> tuple5_conn4 {
  return {tuple_get_src_addr(cn), tuple_get_dst_addr(cn),
          tuple_get_src_port(cn), tuple_get_dst_port(cn), sc};
}

} // namespace SketchVision
