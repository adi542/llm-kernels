NVCC = nvcc
FLAGS = -O3 -arch=sm_75

all: bin/reduction

bin/reduction: src/reduction.cu src/common.cuh
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/reduction src/reduction.cu

clean:
	rm -rf bin

.PHONY: all clean
