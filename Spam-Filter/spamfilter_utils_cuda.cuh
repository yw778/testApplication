#ifndef SPAMFILTER_UTILS_CUDA
#define SPAMFILTER_UTILS_CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "spamfilter_defs.h"
#include "spamfilter_utils.hpp"


// cuda code
// everything with a p_ prefix is done in parallel using CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define checkCuBlasErrors(val) checkBlas( (val), #val, __FILE__, __LINE__)

// check and output CUDA errors
template <typename ErrType>
void check(ErrType err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        printf("Cuda error at %s:%d\n", file, line);
        printf("%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}
// check and output CuBLAS errors
template <typename ErrType>
void checkBlas(ErrType err, const char* const func, const char* const file, const int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("CuBLAS error %d at %s:%d:%s\n", err, file, line, func);
        exit(1);
    }
}

void p_updateParameters(
    cublasHandle_t handle,
    FeatureType* d_theta,
    FeatureType* d_gradient,
    size_t num_feats,
    float step_size,
    bool revert = false);

void p_add_vectors(cublasHandle_t handle, float* a, float* b, const size_t size, const float scale_for_a = 1);

float p_dot_product(cublasHandle_t handle, float* a, float* b, float* d_a, float* d_b, const size_t size);

double p_sigmoid(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_x_i, const size_t num_feats);

#endif
