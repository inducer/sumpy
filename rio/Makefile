CFLAGS=-Wall -W -O2

PATH_TO_NVIDIA_SDK:="/usr/local/cuda_sdk"
PATH_TO_NVIDIA_INC:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/inc
PATH_TO_NVIDIA_LIB:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/lib/Linux64

cpu: tree.c
	g++ $?
	./a.out

gpu: tree.cu
	nvcc $?
	./a.out

kernel: kernel.c
	g++ $(CFLAGS) $?
	./a.out

nbody: nbody.cpp
	g++ $(CFLAGS) -I$(PATH_TO_NVIDIA_INC) $? -L$(PATH_TO_NVIDIA_LIB) -lOpenCL
	./a.out

save:
	rm a.out
	tar zcvf ../test.tgz ../test
