NVCC = nvcc
FLAGS = -O3 -arch=sm_75

all: bin/reduction bin/softmax bin/norms


bin/reduction: src/reduction.cu src/common.cuh
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/reduction src/reduction.cu

bin/softmax: src/softmax.cu src/common.cuh
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/softmax src/softmax.cu

bin/norms: src/norms.cu src/common.cuh src/reduce.cuh
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/norms src/norms.cu

clean:
	rm -rf bin

.PHONY: all clean
