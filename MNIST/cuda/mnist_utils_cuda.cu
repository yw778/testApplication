#include <math.h>

#include "mnist_utils_cuda.cuh"
#include "utils/mnist_timer.h"

// adds two device vectors with CuBLAS and stores the results in the first one
// void p_addVectors(float* a, float* b, const size_t size, const float scale_for_b) {
     // cublasSaxpy(handle, size, &scale_for_a, b, 1, a, 1);
//     }
// }


// void addVectors(float* a, float* b, const size_t size, const float scale_for_b) {
//     for (size_t j = 0; j < size; j++) {
//         a[j] += scale_for_b * b[j];
//     }
// }

// computes dot product with CuBLAS for two given vectors a and b
// float p_dotProduct(float* d_a, float* d_b, const size_t num_elems) {

     // float result[1];
     // cublasSdot (handle, num_elems, d_a, 1, d_b, 1, result);
     // cudaDeviceSynchronize();
     // return *result;
 // }

// float dotProduct(float* d_a, float* d_b, const size_t num_elems) {
//      float result = 0;
//      for (size_t j = 0; j < num_elems; j++) {
//          result += d_a[j] * d_b[j];
//      }
//      return result;
//  }


// Parallel implementation of matrix vector multiplication. Each thread goes
// a certain number of features and strides by the number of threads in the 
// whole mini batch.
__device__ void d_matrixVectorMultiply(
    FeatureType* matrix,
    FeatureType* vect,
    float scalar,
    size_t batch_size,
    size_t num_features,
    size_t threads_per_mini_batch,
    FeatureType* result) {

    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    for (int j = 0; j < batch_size; j++) {
        for (int i = tidx; i < num_features; i += threads_per_mini_batch) {
            // index of the point with respect to the whole dataset
            size_t point_idx = bidx * batch_size + j;
            // index of the feature with respect to all features in the dataset
            size_t feature_idx = point_idx * num_features + i;
            result[i] += matrix[feature_idx] * vect[j] * scalar;
        }
    }
}


// updates the parameters using atomics
__device__ void d_updateParameters(
    FeatureType* gradient,
    FeatureType* parameter_vector,
    size_t num_features,
    size_t threads_per_mini_batch,
    double step_size) {

    size_t tidx = threadIdx.x;
    
    for (size_t i = tidx; i < num_features; i += threads_per_mini_batch) {
        FeatureType gradient_times_step_size = gradient[i] * step_size;
        atomicAdd(&parameter_vector[i], -gradient_times_step_size);
    }
}


// initializes all values in array to a certain value
__device__ void d_memset(
    FeatureType* array,
    float value,
    size_t num_elements,
    size_t threads_per_mini_batch) {

    size_t tidx = threadIdx.x;
    for (size_t i = tidx; i < num_elements; i += threads_per_mini_batch) {
        array[i] = value;
    }
}


// computes logistic function for a given parameter vector (theta) and a data point (x_i)
// double p_logisticFunction(FeatureType* d_theta, FeatureType* d_x_i, const size_t num_features) {
//      return logisticFunction(p_dotProduct(d_theta, d_x_i, num_features));
// }


// double logisticFunction(FeatureType* d_theta, FeatureType* d_x_i, const size_t num_features) {
    // return d_logisticFunction(dotProduct(d_theta, d_x_i, num_features));
//}


// computes logistic function with fast exp
__device__ float d_logisticFunction(float exponent) {
    return (1.0f / (1.0f + __expf(-exponent)));
}

// verify the device properties satisfy the assumptions of the kernel
// check that the resulting grid and block dimensions
// dont' violate device limits
bool checkDeviceProps(
    size_t shared_memory_size,
    dim3 block_size,
    dim3 grid_size) {

    bool devicePropsOK = true;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (shared_memory_size > deviceProp.sharedMemPerBlock) {

        // printf("Shared Memory size exceeds maximum allowed size.\n");
        printf("SM-");
        devicePropsOK = false;
    }

    if (block_size.x > deviceProp.maxThreadsDim[0]
        || grid_size.x > deviceProp.maxGridSize[0]) {

        // printf("Grid or block size exceeds maximum allowed size.\n");
        printf("B-");
        devicePropsOK = false;
    }

    return devicePropsOK;
}

// updates the parameters (theta)
// void p_updateParameters(FeatureType* d_theta, FeatureType* d_gradient, size_t num_features, float step_size, bool revert) {
     // float sign = revert ? 1 : -1;
     // step_size *= sign;
     // addVectors(d_theta, d_gradient, num_features, step_size);
     // cublasSaxpy(handle, num_features, &step_size, d_gradient, 1, d_theta, 1);
//}
