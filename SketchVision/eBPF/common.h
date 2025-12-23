#ifndef __COMMON_H
#define __COMMON_H

#include <linux/types.h>

// #define IMAGE_SIZE 1024
// #define IMAGE_SIZE_D 1048576
#define IMAGE_SIZE 2048
#define IMAGE_SIZE_D 4194304

#define MAX_FLOWS 65536

#define ANGLE_DIVISIONS 3600
#define LUT_SIZE 1000000
#define SCALE_12 1000000000000
#define SCALE_6 1000000
// #define SCALE_12 10000

#define TWO_PI_SCALED 6283185ULL // 2π × 1e6
#define SIN_LUT_SIZE 2048        // Or 2048, etc.
#define SCALE 128                // 1e6 fixed-point scale
#define MAX_SQRT_VAL 6283185     // 2π * 1e6, because your values are scaled
#define ANGLE_DIVISOR 628

#define HWINDOW 32
#define WINDOW 64

struct event_t {
  __s32 x;
  __s32 y;
  __s32 cnt;
};

struct flow_key_t {
  __u32 src_ip;
  __u32 dst_ip;
  //__u16 src_port;
  // __u16 dst_port;
  // __u8 proto;
};

struct flow_stat_t {
  __u32 pkt_count;
  __u64 t0;
};

struct pixel_t {
  __u8 r, g, b;
} __attribute__((packed));

#endif
