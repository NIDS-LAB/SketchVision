#include "common.h"
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/if_link.h>
#include <math.h>
#include <net/if.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define HEIGHT 64
#define WIDTH 64

#define PIN_PATH "/sys/fs/bpf/image"

struct pixel_t *mmap_image_map(int map_fd) {
  size_t map_size = IMAGE_SIZE_D * sizeof(struct pixel_t);
  void *addr = mmap(NULL, map_size, PROT_READ | PROT_READ,
                    MAP_SHARED | MAP_SYNC, map_fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }
  return (struct pixel_t *)addr;
}

void write_ppm(const char *filename, __u8 image[HEIGHT][WIDTH][3]) {
  printf("1-Going to open file: %s", filename);
  FILE *f = fopen(filename, "wb");
  printf("2-Opened file: %s", filename);
  fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
  fwrite(image, 1, WIDTH * HEIGHT * 3, f);
  fclose(f);
}

// Custom print function to print all levels of libbpf messages
static int my_libbpf_verbose_printer(enum libbpf_print_level level,
                                     const char *format, va_list args) {
  // You can filter by level if you want, e.g.:
  // if (level > LIBBPF_INFO) // Suppress debug messages
  //     return 0;
  return vfprintf(stderr, format, args); // Print all levels to stderr
}

struct pixel_t *image_base = NULL;
int map_array;
int handle_event(void *ctx, void *data, size_t data_sz) {
  struct event_t *e = (struct event_t *)data;

  int32_t half_box = 64 / 2;
  __u8 img[HEIGHT][WIDTH][3];
  for (int dy = -half_box; dy < half_box; dy++) {
    for (int dx = -half_box; dx < half_box; dx++) {
      int32_t px = ((int32_t)e->x + dx + IMAGE_SIZE) % IMAGE_SIZE;
      int32_t py = ((int32_t)e->y + dy + IMAGE_SIZE) % IMAGE_SIZE;

      atomic_thread_fence(memory_order_acquire);
      __u32 index = py * IMAGE_SIZE + px;
      struct pixel_t pixel; // = image_base[index];
      bpf_map_lookup_elem(map_array, &index, &pixel);
      int iy = half_box + dy;
      int ix = half_box + dx;
      img[iy][ix][0] = pixel.r;
      img[iy][ix][1] = pixel.g;
      img[iy][ix][2] = pixel.b;
    }
  }
  write_ppm("./img.ppm", img);
  printf("Completed Snapshot: X=%u, Y=%u, pkt_cnt=%u\n", e->x, e->y, e->cnt);

  return 0;
}

void fill_lookup_tables(int map_sin, int map_cos, int map_sqrt) {

  for (int i = 0; i < SIN_LUT_SIZE; i++) {
    // Compute angle in radians from 0 to 2π
    double angle = i;
    int32_t sin_val = (int32_t)(sin(angle) * SCALE);
    int32_t cos_val = (int32_t)(cos(angle) * SCALE);
    uint32_t idx = i;
    bpf_map_update_elem(map_sin, &idx, &sin_val, BPF_ANY);
    bpf_map_update_elem(map_cos, &idx, &cos_val, BPF_ANY);
  }

  for (int i = 0; i < LUT_SIZE; i++) {
    double val = sqrt((double)i);
    __u32 sqrt_val = (__u32)(val); // e.g., fixed-point with 2 decimal places
    bpf_map_update_elem(map_sqrt, &i, &sqrt_val, BPF_ANY);
  }
}

void unpin_map(const char *path) {
  if (remove(path) < 0 && errno != ENOENT) {
    fprintf(stderr, "Warning: could not remove pinned map at %s: %s\n", path,
            strerror(errno));
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <ifname>\n", argv[0]);
    return 1;
  }

  const char *ifname = argv[1];
  int ifindex = if_nametoindex(ifname);
  if (ifindex == 0) {
    perror("if_nametoindex");
    return 1;
  }

  libbpf_set_strict_mode(LIBBPF_STRICT_ALL); // Optional, for stricter checking
  libbpf_set_print(my_libbpf_verbose_printer);

  struct bpf_object *obj;
  int prog_fd;
  obj = bpf_object__open_file("xdp_prog_kern.o", NULL);
  if (libbpf_get_error(obj)) {
    fprintf(stderr, "Failed to open BPF object\n");
    return 1;
  }

  if (bpf_object__load(obj)) {
    fprintf(stderr, "Failed to load BPF object\n");
    return 1;
  }

  // 📌 GET MAP FDs
  int map_sin = bpf_object__find_map_fd_by_name(obj, "sin_lut");
  int map_cos = bpf_object__find_map_fd_by_name(obj, "cos_lut");
  int map_sqrt = bpf_object__find_map_fd_by_name(obj, "sqrt_lut");
  map_array = bpf_object__find_map_fd_by_name(obj, "image");
  int map_event = bpf_object__find_map_fd_by_name(obj, "events");

  if (map_sin < 0 || map_cos < 0 || map_sqrt < 0) {
    fprintf(stderr, "Failed to find one or more LUT maps\n");
    return 1;
  }

  // 🧠 FILL THE TABLES HERE
  fill_lookup_tables(map_sin, map_cos, map_sqrt);

  int map_fd = bpf_obj_get(PIN_PATH);
  if (map_fd < 0) {
    fprintf(stderr, "ERROR: Failed to open pinned map at %s: %s\n", PIN_PATH,
            strerror(errno));
    return 1;
  }
  image_base = mmap_image_map(map_fd);

  struct bpf_program *prog = bpf_object__find_program_by_title(obj, "xdp");
  if (!prog) {
    fprintf(stderr, "Failed to find xdp program\n");
    return 1;
  }

  prog_fd = bpf_program__fd(prog);
  if (prog_fd < 0) {
    fprintf(stderr, "Failed to get program fd\n");
    return 1;
  }

  int xdp_flags = XDP_FLAGS_SKB_MODE;
  if (bpf_set_link_xdp_fd(ifindex, prog_fd, xdp_flags) < 0) {
    perror("bpf_set_link_xdp_fd");
    return 1;
  }

  printf("XDP program loaded and attached to %s\n", ifname);

  struct ring_buffer *rb =
      ring_buffer__new(map_event, handle_event, NULL, NULL);

  while (1) {
    // sleep(1);
    ring_buffer__poll(rb, 100 /* ms timeout */);
  }

  return 0;
}
