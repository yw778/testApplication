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
