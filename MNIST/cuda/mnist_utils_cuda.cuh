#ifndef MNIST_UTILS_CUDA
#define MNIST_UTILS_CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "utils/mnist_defs.h"
#include "utils/mnist_utils.hpp"


// cuda code
// everything with a p_ prefix is done in parallel using CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// #include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
// #define checkCuBlasErrors(val) checkBlas( (val), #val, __FILE__, __LINE__)
#define DIVIDE_AND_CEIL(obj_dim, block_dim) ((obj_dim + block_dim - 1) / block_dim)
#define GET_GLOBAL_THREAD_INDEX() (blockIdx.x * blockDim.x + threadIdx.x)

// check and output CUDA errors
template <typename ErrType>
void check(ErrType err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        printf("Cuda error at %s:%d\n", file, line);
        printf("%s -- from function %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}
// check and output CuBLAS errors
// template <typename ErrType>
// void checkBlas(ErrType err, const char* const func, const char* const file, const int line) {
//     if (err != CUBLAS_STATUS_SUCCESS) {
//         printf("CuBLAS error %d at %s:%d:%s\n", err, file, line, func);
//         exit(1);
//     }
// }

//loads the contents of a global memory vector into shared memory
template <typename Type>
__device__ void loadSharedVector(size_t num_elems, Type* d_vector, Type* shared_vector) {
    size_t iterations = DIVIDE_AND_CEIL(num_elems, blockDim.x);
    for (size_t i = 0; i < iterations; i++) {
        size_t index = blockDim.x * i + threadIdx.x;
        if (index < num_elems) {
            shared_vector[index] = d_vector[index];
        }
    }
}

//downloads the contents of shared memory into a global memory vector
template <typename Type>
__device__ void downloadSharedVector(size_t num_elems, Type* shared_vector, Type* d_vector) {
    size_t iterations = DIVIDE_AND_CEIL(num_elems, blockDim.x);
    for (size_t i = 0; i < iterations; i++) {
        size_t index = blockDim.x * i + threadIdx.x;
        if (index < num_elems) {
            d_vector[index] = shared_vector[index];
        }
    }
}

// verify the device properties satisfy the assumptions of the kernel
// check that the resulting grid and block dimensions
// dont' violate device limits
bool checkDeviceProps(
    size_t shared_memory_size,
    dim3 block_size,
    dim3 grid_size);

__device__ void d_memset(
    FeatureType* array,
    float value,
    size_t num_elements,
    size_t threads_per_mini_batch);

// void p_updateParameters(
//      cublasHandle_t handle,
//      FeatureType* d_theta,
//      FeatureType* d_gradient,
//      size_t num_features,
//      float step_size,
//      bool revert = false);

// updates the parameters using atomics 
__device__ void d_updateParameters(
    FeatureType* gradient,
    FeatureType* parameter_vector,
    size_t num_features,
    size_t threads_per_mini_batch,
    double step_size);

// void p_add_vectors(cublasHandle_t handle, float* a, float* b, const size_t size, const float scale_for_a = 1);

// void addVectors(float* a, float* b, const size_t size, const float scale_for_b);

// float p_dotProduct(cublasHandle_t handle, float* a, float* b, float* d_a, float* d_b, const size_t size);

// void p_MatrixVectorMultiply(cublasHandle_t handle, float* matrix, float* vect, float scalar, size_t num_data_points, size_t num_features, float* result);

// Parallel implementation of matrix vector multiplication. Each thread goes
// a certain number of features and strides by the number of threads in the 
// whole mini batch.
__device__ void d_matrixTranspose(
    FeatureType* probility_matrix,
    FeatureType* probility_transpose,
    size_t batch_size,
    size_t relative_tidx,
    size_t point_idx_in_block);

__device__ void d_matrixTranspose2(
    FeatureType* probility_matrix,
    FeatureType* probility_transpose,
    size_t batch_size);


__device__ void d_matrixMatrixMultiply(
    FeatureType* probility_matrix,
    FeatureType* datapoint_matrix,
    float scalar,
    size_t batch_size,
    size_t num_features,
    size_t threads_per_mini_batch,
    FeatureType* result);



__device__ void d_matrixVectorMultiply(
    FeatureType* matrix,
    FeatureType* vect,
    float scalar,
    size_t batch_size,
    size_t num_features,
    size_t threads_per_mini_batch,
    FeatureType* result);

// double p_logisticFunction(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_x_i, const size_t num_features);
__device__ float d_logisticFunction(float exponent);



__device__ void  d_softMaxFunction1(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t point_idx_in_shmem,
    size_t relative_tidx,
    size_t point_idx_in_block,
    size_t num_label);


__device__ void  d_softMaxFunction2(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t point_idx_in_shmem,
    size_t relative_tidx,
    size_t point_idx_in_block,
    size_t num_thread_each_label);

__device__ void  d_softMaxFunction3(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t relative_tidx,
    size_t point_idx_in_batch,
    size_t num_label);

__device__ void  d_softMaxFunction4(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t relative_tidx,
    size_t point_idx_in_batch,
    size_t num_label);

#endif