
# COMPILER = g++
# CUDA_COMPILER = nvcc
# BIN_DIR = bin
# CUDA_INCLUDEPATH=/usr/local/cuda-7.0/include
# BLAS_INSTALL_DIR=/opt/OpenBLAS
# BLAS_INCLUDEANDLINK=-I $(BLAS_INSTALL_DIR)/include -L$(BLAS_INSTALL_DIR)/lib -lopenblas
# CUDA_FLAGS=-O3 -arch=sm_35 -Xcompiler -Wall -Xcompiler -Wextra -m64 -lcublas
# C_FLAGS=-O3 -Wall -Wextra -m64 -pedantic -ansi

COMPILER = g++
CUDA_COMPILER = nvcc

BIN_DIR = bin
CUDA_DIR=/usr/local/cuda-7.0/
BLAS_INSTALL_DIR=/usr/lib/
CUBLAS_INSTALL_DIR=$(CUDA_DIR)targets/x86_64-linux/

CUDA_INCLUDEPATH=$(CUDA_DIR)include


CUBLAS_INCLUDEANDLINK=-I $(CUBLAS_INSTALL_DIR)include -L$(CUBLAS_INSTALL_DIR)lib -lcublas
BLAS_INCLUDEANDLINK=-I $(BLAS_INSTALL_DIR)include -L$(BLAS_INSTALL_DIR)lib -lopenblas
CUDA_FLAGS=-O3 -arch=sm_35 -Xcompiler -Wall -Xcompiler -Wextra -m64
C_FLAGS=-O3 -Wall -Wextra -m64


all: spam-filter

spam-filter: main.o baseline.o batch_baseline.o alt_one.o spamfilter_timer.o spamfilter_utils.o spamfilter_utils_cuda.o
	$(CUDA_COMPILER) $^ -o bin/main $(CUDA_FLAGS) $(BLAS_INCLUDEANDLINK) $(CUBLAS_INCLUDEANDLINK)

baseline-only: main.o baseline.o batch_baseline.o spamfilter_timer.o spamfilter_utils.o
	$(COMPILER) $^ -o bin/main $(C_FLAGS) $(BLAS_INCLUDEANDLINK) $(CUBLAS_INCLUDEANDLINK)

main.o: main.cpp
	$(COMPILER) -c $^ -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

baseline.o: baseline.cpp
	$(COMPILER) -c $^ -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

batch_baseline.o: batch_baseline.cpp
	$(COMPILER) -c $^ -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

alt_one.o: alt_one.cu
	$(CUDA_COMPILER) -c $^ $(CUDA_FLAGS) $(CUBLAS_INCLUDEANDLINK)

spamfilter_timer.o: spamfilter_timer.cpp
	$(COMPILER) -c $^ -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

spamfilter_utils.o: spamfilter_utils.cpp
	$(COMPILER) -c $^ -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

spamfilter_utils_cuda.o: spamfilter_utils_cuda.cu
	$(CUDA_COMPILER) -c $^ $(CUDA_FLAGS) $(CUBLAS_INCLUDEANDLINK)

clean:
	rm -f *.o $(BIN_DIR)/main parameter_vector*
