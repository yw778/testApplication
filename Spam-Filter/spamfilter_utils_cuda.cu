#include <math.h>

#include "spamfilter_utils_cuda.cuh"
#include "spamfilter_timer.h"

// adds two device vectors with CuBLAS and stores the results in the first one
void p_add_vectors(cublasHandle_t handle, float* a, float* b, const size_t size, const float scale_for_a) {
    cublasSaxpy(handle, size, &scale_for_a, b, 1, a, 1);
}

// computes dot product with CuBLAS for two given vectors a and b
float p_dot_product(cublasHandle_t handle, float* d_a, float* d_b, const size_t num_elems) {

    float result[1];
    cublasSdot (handle, num_elems, d_a, 1, d_b, 1, result);
    cudaDeviceSynchronize();
    return *result;
}


// computes logistic function for a given parameter vector (theta) and a data point (x_i)
double p_sigmoid(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_x_i, const size_t num_feats) {
    return sigmoid(p_dot_product(handle, d_theta, d_x_i, num_feats));
}


// updates the parameters (theta)
void p_updateParameters(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_gradient, size_t num_feats, float step_size, bool revert) {
    float sign = revert ? 1 : -1;
    step_size *= sign;
    cublasSaxpy(handle, num_feats, &step_size, d_gradient, 1, d_theta, 1);
}
