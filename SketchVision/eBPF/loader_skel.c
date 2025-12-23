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

#include "xdp_prog_kern.skel.h"

#define HEIGHT 64
#define WIDTH 64

#define PIN_PATH "/sys/fs/bpf/my_image_map"

struct pixel_t *mmap_image_map(int map_fd) {
  size_t map_size = IMAGE_SIZE * IMAGE_SIZE * sizeof(struct pixel_t);
  void *addr =
      mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, map_fd, 0);
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

  printf("Got one!\n");

  __u32 index = e->y * IMAGE_SIZE + e->x;
  struct pixel_t pixel = image_base[index];
  printf("(R:%d, G:%d, B:%d) (X: %d,Y: %d)\n", pixel.r, pixel.g, pixel.b, e->x,
         e->y);

  return 0;
  int32_t half_box = 64 / 2;
  __u8 img[HEIGHT][WIDTH][3];
  for (int dy = -half_box; dy < half_box; dy++) {
    for (int dx = -half_box; dx < half_box; dx++) {
      int32_t px = ((int32_t)e->x + dx + IMAGE_SIZE) % IMAGE_SIZE;
      int32_t py = ((int32_t)e->y + dy + IMAGE_SIZE) % IMAGE_SIZE;

      // printf("(%d, %d)\n", px, py);

      atomic_thread_fence(memory_order_acquire);
      __u32 index = py * IMAGE_SIZE + px;
      struct pixel_t pixel = image_base[index];
      int iy = half_box + dy;
      int ix = half_box + dx;
      img[iy][ix][0] = pixel.r;
      img[iy][ix][1] = pixel.g;
      img[iy][ix][2] = pixel.b;
      if (pixel.b != 0 || pixel.r != 0 || pixel.g != 0)
        printf("(R:%d, G:%d, B:%d) (X: %d,Y: %d)", pixel.r, pixel.g, pixel.b,
               py, px);
      struct pixel_t pix;
      if (px == 655 && py == 4006 &&
          bpf_map_lookup_elem(map_array, &index, &pix) == 0) {
        printf("$$ (R:%d, G:%d, B:%d)\n", pix.r, pix.g, pix.b);
        printf("(R:%d, G:%d, B:%d: %u)", pixel.r, pixel.g, pixel.b, index);
      } else if (px == 655 && py == 4006) {
        perror("bpf_map_lookup_elem failed");
        printf("Failed to read index %u\n, %d, %d", index, py, px);
      }
    }
  }
  write_ppm("./img.ppm", img);
  printf("Completed Snapshot: X=%u, Y=%u, pkt_cnt=%u\n", e->x, e->y, e->cnt);

  return 0;
}

void fill_lookup_tables(int map_sin, int map_cos, int map_sqrt) {

  for (int i = 0; i < SIN_LUT_SIZE; i++) {
    // Compute angle in radians from 0 to 2π
    double angle = (2.0 * M_PI * i) / (double)SIN_LUT_SIZE;
    int32_t sin_val = (int32_t)(sin(angle) * SCALE);
    int32_t cos_val = (int32_t)(cos(angle) * SCALE);
    uint32_t idx = i;
    bpf_map_update_elem(map_sin, &idx, &sin_val, BPF_ANY);
    bpf_map_update_elem(map_cos, &idx, &cos_val, BPF_ANY);
  }

  for (int i = 0; i < LUT_SIZE; i++) {
    double val = sqrt((double)i);
    __u16 sqrt_val =
        (__u16)(val * SCALE); // e.g., fixed-point with 2 decimal places
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

  libbpf_set_strict_mode(LIBBPF_STRICT_ALL); // Optional, for stricter
  // checking
  libbpf_set_print(my_libbpf_verbose_printer);

  struct xdp_prog_kern *skel = xdp_prog_kern__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open/load skeleton\n");
    return 1;
  }
  printf("✅ XDP program loaded \n");

  // 📌 GET MAP FDs
  int map_sin = bpf_map__fd(skel->maps.sin_lut);
  int map_cos = bpf_map__fd(skel->maps.cos_lut);
  int map_sqrt = bpf_map__fd(skel->maps.sqrt_lut);
  map_array = bpf_map__fd(skel->maps.image);
  int map_event = bpf_map__fd(skel->maps.events);

  if (map_sin < 0 || map_cos < 0 || map_sqrt < 0) {
    fprintf(stderr, "Failed to find one or more LUT maps\n");
    return 1;
  }

  // bpf_map__unpin(skel->maps.image, PIN_PATH); // optional cleanup
  // if (bpf_map__pin(skel->maps.image, PIN_PATH)) {
  //   fprintf(stderr, "Failed to pin map at %s\n", PIN_PATH);
  //  goto cleanup;
  //}

  // 🧠 FILL THE TABLES HERE
  fill_lookup_tables(map_sin, map_cos, map_sqrt);

  struct bpf_program *prog = skel->progs.xdp_prog_main;
  if (!prog) {
    fprintf(stderr, "Failed to find program 'xdp'\n");
    return 1;
  }

  int prog_fd = bpf_program__fd(prog);
  if (prog_fd < 0) {
    fprintf(stderr, "❌ Failed to get program FD\n");
    return 1;
  }

  // if (xdp_prog_kern__attach(skel)) {
  //   fprintf(stderr, "Failed to attach XDP program\n");
  //   goto cleanup;
  // }
  printf("XDP program loaded and attached to %s\n", ifname);

  struct ring_buffer *rb =
      ring_buffer__new(map_event, handle_event, NULL, NULL);

  int map_fd = bpf_map__fd(skel->maps.image);
  size_t map_size = IMAGE_SIZE_D * sizeof(struct pixel_t);
  void *map_data =
      mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, map_fd, 0);
  if (map_data == MAP_FAILED) {
    perror("mmap failed");
    goto cleanup;
  }

  int xdp_flags = XDP_FLAGS_SKB_MODE;
  if (bpf_set_link_xdp_fd(ifindex, prog_fd, xdp_flags) < 0) {
    perror("bpf_set_link_xdp_fd");
    return 1;
  }

  image_base = (struct pixel_t *)map_data;
  char *tmp = map_data;
  for (size_t i = 0; i < 16; i++) {
    printf("%02x ", tmp[i]);
  }

  while (1) {
    // sleep(1);
    ring_buffer__poll(rb, 100 /* ms timeout */);
  }

  return 0;

cleanup:
  xdp_prog_kern__destroy(skel);
}
