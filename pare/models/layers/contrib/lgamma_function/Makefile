OBJ = $(patsubst %.cu,%.o,$(wildcard src/*.cu))

default:
	python build.py

cuda: $(OBJ)
	python build.py -with_cuda

%.o: %.cu
	nvcc -c -o $@ $< -x cu -Xcompiler -fPIC -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
