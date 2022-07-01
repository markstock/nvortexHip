HIPCC=hipcc
# the MI250X
ARCH=--offload-arch=gfx90a
# the MI100
#ARCH=--offload-arch=gfx908
HIPCCFLAGS=-O2 -ffast-math
#HIPCCFLAGS=-O2
CXXFLAGS=-march=native -fopenmp -fomit-frame-pointer
# we need this to prevent skipping Kahan summation altogether
KAHANFLAGS=-fno-associative-math

all : nvHip01.bin nvHip02.bin nvHip03.bin nvHip04.bin nvHip05.bin ngHip05.bin ngHip06.bin

%.bin : %.hip
	$(HIPCC) $(ARCH) $(HIPCCFLAGS) $(CXXFLAGS) -o $@ $^

nvHip04.bin : nvHip04.hip
	$(HIPCC) $(ARCH) $(HIPCCFLAGS) $(KAHANFLAGS) $(CXXFLAGS) -o $@ $^

clean :
	rm nvHip01.bin nvHip02.bin nvHip03.bin nvHip04.bin nvHip05.bin ngHip05.bin ngHip06.bin
