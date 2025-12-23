CLANG=clang
BPFTOOL=bpftool

CFLAGS=-O2 -g -Wall -target bpf  -D__TARGET_ARCH_x86
LIBBPF_OBJ=-O2 -lbpf -lelf -lz

all: xdp_prog_kern.o user_loader

xdp_prog_kern.o: xdp_prog.c common.h
	$(CLANG) $(CFLAGS) -c xdp_prog.c -o xdp_prog_kern.o

user_loader: loader.c common.h
	bpftool gen skeleton xdp_prog_kern.o > xdp_prog_kern.skel.h
	gcc -g -Wall -O2 loader.c -o user_loader -lm $(LIBBPF_OBJ)

#user_loader: loader_skel.c common.h
#	bpftool gen skeleton xdp_prog_kern.o > xdp_prog_kern.skel.h
#	gcc -g -Wall -O2 loader_skel.c -o user_loader -lm $(LIBBPF_OBJ)

clean:
	rm -f *.o user_loader



#bpftool gen skeleton xdp_prog_kern.o > mmap.skel.h
