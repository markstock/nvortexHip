HIPCC=hipcc
# the MI250X
ARCH=--offload-arch=gfx90a
# the MI100
#ARCH=--offload-arch=gfx908
# -Ofast is not as good as -O2
HIPCCFLAGS=-O2 -ffast-math -flto
CXXFLAGS=-march=native -fopenmp -fomit-frame-pointer
# we need this to prevent skipping Kahan summation altogether
KAHANFLAGS=-fno-associative-math

BINS=nvHip01.bin nvHip02.bin nvHip03.bin nvHip04.bin nvHip05.bin ngHip05.bin ngHip06.bin ngHip07.bin

all : $(BINS)

%.bin : %.hip
	$(HIPCC) $(ARCH) $(HIPCCFLAGS) $(CXXFLAGS) -o $@ $^

nvHip04.bin : nvHip04.hip
	$(HIPCC) $(ARCH) $(HIPCCFLAGS) $(KAHANFLAGS) $(CXXFLAGS) -o $@ $^

clean :
	rm $(BINS)
