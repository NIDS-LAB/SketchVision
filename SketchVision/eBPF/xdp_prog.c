#include "common.h"
#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <math.h>

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 14); // 16MB buffer
} events SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, MAX_FLOWS);
  __type(key, struct flow_key_t);
  __type(value, struct flow_stat_t);
} flow_stats SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, IMAGE_SIZE_D);
  __type(key, __u32);
  __type(value, struct pixel_t);
  __uint(map_flags, BPF_F_MMAPABLE);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} image SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, LUT_SIZE);
  __type(key, __u32);
  __type(value, __u32);
} sqrt_lut SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, ANGLE_DIVISIONS);
  __type(key, __u32);
  __type(value, __s32);
} sin_lut SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, ANGLE_DIVISIONS);
  __type(key, __u32);
  __type(value, __s32);
} cos_lut SEC(".maps");

static __always_inline __u8 reverse_bits(__u8 n) {
  __u8 r = 0;
  for (int i = 0; i < 6; i++) {
    r <<= 1;
    r |= (n & 1);
    n >>= 1;
  }
  return r;
}

static __always_inline __u32 hash_fn(__u32 ip1, __u32 ip2,
                                     __u32 seed) { // MurmurHash3 finalizer
  __u32 h = ip1 ^ ip2 ^ seed;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

#ifndef bpf_htonl
#define bpf_htonl(x) __builtin_bswap32(x)
#endif

SEC("xdp")
int xdp_prog_main(struct xdp_md *ctx) {
  void *data_end = (void *)(long)ctx->data_end;
  void *data = (void *)(long)ctx->data;

  __u64 ts_ = bpf_ktime_get_ns();
  bpf_printk("Arrived: Got packet, time: %llu", ts_);

  __u64 pkt_size = (__u64)((__u8 *)data_end - (__u8 *)data);

  struct ethhdr *eth = data;

  if ((void *)(eth + 1) > data_end)
    return XDP_PASS;

  if (eth->h_proto != __constant_htons(ETH_P_IP))
    return XDP_PASS;

  struct iphdr *ip = data + sizeof(struct ethhdr);
  if ((void *)(ip + 1) > data_end)
    return XDP_PASS;

  if (ip->protocol != IPPROTO_TCP && ip->protocol != IPPROTO_UDP)
    return XDP_PASS;

  __u16 sport = 0, dport = 0;
  __u64 ts = 0;
  if (ip->protocol == IPPROTO_TCP) {
    struct tcphdr *tcp = (void *)ip + ip->ihl * 4;
    if ((void *)(tcp + 1) > data_end)
      return XDP_PASS;
    sport = tcp->source;
    dport = tcp->dest;
    ts = __builtin_bswap32(tcp->seq);
  } else {
    struct udphdr *udp = (void *)ip + ip->ihl * 4;
    if ((void *)(udp + 1) > data_end)
      return XDP_PASS;
    sport = udp->source;
    dport = udp->dest;
  }

  // bpf_printk("Finished parsing: Got packet, time: %llu", ts);

  struct flow_key_t key = {};
  key.src_ip = ip->saddr;
  key.dst_ip = ip->daddr;
  // key.src_port = sport;
  // key.dst_port = dport;
  // key.proto = ip->protocol;

  struct flow_stat_t *stat =
      (struct flow_stat_t *)bpf_map_lookup_elem(&flow_stats, &key);
  struct flow_stat_t new_stat = {};
  __u64 ipd = 0;
  if (stat) {
    // ipd = ts - stat->last_ts;
    // stat->ipd_sum += ipd;
    stat->pkt_count += 1;
    // stat->last_ts = ts;
  } else {
    // new_stat.last_ts = ts;
    // new_stat.ipd_sum = 0;
    new_stat.pkt_count = 1;
    new_stat.t0 = ts;
    bpf_map_update_elem(&flow_stats, &key, &new_stat, BPF_ANY);
    stat = &new_stat;
    /// return XDP_PASS;
  }

  // bpf_printk("Before Encoding: Got packet, time: %llu", ts);

  __u32 time_ms = (ts - stat->t0) / 1000;
  if (time_ms >= LUT_SIZE)
    time_ms = LUT_SIZE - 1;
  __u32 *sqrt_val = (__u32 *)bpf_map_lookup_elem(&sqrt_lut, &time_ms);
  if (!sqrt_val)
    return XDP_PASS;

  __u32 angle_idx = (*sqrt_val); // already fixed point, e.g., 1e6 scale

  if (angle_idx >= SIN_LUT_SIZE)
    angle_idx = SIN_LUT_SIZE - 1;

  __s32 *sin_val = (__s32 *)bpf_map_lookup_elem(&sin_lut, &angle_idx);
  __s32 *cos_val = (__s32 *)bpf_map_lookup_elem(&cos_lut, &angle_idx);
  if (!sin_val || !cos_val)
    return XDP_PASS;

  __u32 sum_div = (time_ms / stat->pkt_count) / 2;
  if (sum_div >= LUT_SIZE)
    sum_div = LUT_SIZE - 1;
  __u32 *radius = (__u32 *)bpf_map_lookup_elem(&sqrt_lut, &sum_div);
  if (!radius)
    return XDP_PASS;

  __s32 x = hash_fn(ip->saddr, ip->daddr, 937529) % IMAGE_SIZE;
  __s32 y = hash_fn(ip->saddr, ip->daddr, 359873) % IMAGE_SIZE;

  __u32 r = *radius;
  __s64 rcos = ((*cos_val) * (__s64)r);
  __s64 rsin = ((*sin_val) * (__s64)r);
  rcos = rcos >> 7;
  rsin = rsin >> 7;
  //  r = r >> 10;

  __s32 p_x = (HWINDOW + rcos) & (WINDOW - 1);
  __s32 p_y = (HWINDOW + rsin) & (WINDOW - 1);

  // bpf_printk("END: initial. (px, py): (%d, %d)", p_x, p_y);

  p_x = (x + p_x + IMAGE_SIZE - HWINDOW) & (IMAGE_SIZE - 1);
  p_y = (y + p_y + IMAGE_SIZE - HWINDOW) & (IMAGE_SIZE - 1);

  // bpf_printk("END: Ready for (px, py): (%d, %d), %lld", p_x, p_y,
  //           stat->pkt_count);
  // bpf_printk("END: Ready for (t_ms, angle): (%ld, %ld)", time_ms, angle_idx);
  // bpf_printk("END: Ready for (rcos, rsin): (%lld, %lld)", rcos, rsin);

  __u32 pixel_index = p_y * IMAGE_SIZE + p_x;
  struct pixel_t *pix = bpf_map_lookup_elem(&image, &pixel_index);
  if (!pix)
    return XDP_PASS;

  // bpf_printk("END: Got pix (px, py): (%d, %d)", p_x, p_y);

  __u16 col_pix = (reverse_bits(pix->g & 0x3F) << 8) | pix->r;
  __u16 val1 = pkt_size;

  if (col_pix == 0) {
    pix->b = val1 & 0xFF;
    pix->g = ((val1 >> 8) & 0x03) << 6;
    pix->r = 8;
  } else if (col_pix < 2040) {
    __u16 b_pkts = (((pix->g & 0xC0) >> 6) << 8) | pix->b;
    b_pkts = (b_pkts * (col_pix / 8) + val1) / (col_pix / 8 + 1);
    pix->b = b_pkts & 0xFF;
    col_pix += 8;
    pix->r = col_pix & 0xFF;
    pix->g = ((b_pkts >> 8) & 0x03) << 6 | reverse_bits((col_pix >> 8) & 0x3F);
  }

  // bpf_printk("END: Done updaiting pix (px, py): (%d, %d)", p_x, p_y);

  if (stat->pkt_count >= 40 && stat->pkt_count % 5 == 0) {
    struct event_t *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
      return XDP_PASS;
    e->x = x;
    e->y = y;
    e->cnt = stat->pkt_count;
    bpf_ringbuf_submit(e, 0);
  }
  // ts = bpf_ktime_get_ns();
  bpf_printk("END: Center. (x, y): (%d, %d)", x, y);
  bpf_printk("END: Got packet. (x, y): (%d, %d)", p_x, p_y);
  bpf_printk("END: Got packet. (R, G, B): (%d, %d, %d)", pix->r, pix->g,
             pix->b);
  bpf_printk("index = %d", pixel_index);
  return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
