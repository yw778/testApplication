#include <stdio.h>
#include <string.h>

#include <cublas_v2.h>

#include "alt_one.h"
#include "spamfilter_utils_cuda.cuh"


static float *d_theta, *d_X, *d_gradient;
static cublasHandle_t handle;

static void setCudaVariables(size_t num_feats, size_t num_points, FeatureType* X) {

    checkCuBlasErrors(cublasCreate(&handle));

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    const size_t vector_size = num_feats * sizeof(FeatureType);

    checkCudaErrors(cudaMalloc(&d_theta, vector_size));
    checkCudaErrors(cudaMalloc(&d_gradient, vector_size));
    checkCudaErrors(cudaMalloc(&d_X, vector_size * num_points));

    checkCudaErrors(cudaMemcpy(d_X, X, vector_size * num_points, cudaMemcpyHostToDevice));
}

static void cleanUp() {

    checkCudaErrors(cudaFree(d_theta));
    checkCudaErrors(cudaFree(d_gradient));
    checkCudaErrors(cudaFree(d_X));

    cublasDestroy(handle);
}

// computes gradient for a single datapoint using cublas
static void p_gradientForSinglePoint(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_x_i, LabelType y, size_t num_feats, FeatureType* d_gradient) {
    float probability_of_positive = p_sigmoid(handle, d_theta, d_x_i, num_feats);
    float pi_minus_yi = probability_of_positive - y;
    checkCudaErrors(cudaMemset(d_gradient, 0, num_feats * sizeof(FeatureType)));
    p_add_vectors(handle, d_gradient, d_x_i, num_feats, pi_minus_yi);
}

// executes serial implementation of stochastic gradient descent for logistic regression with a fixed number of iterations
void trainParallelStochasticGradientDescent1(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs,
    double tolerance,
    double step_size,
    size_t num_points,
    size_t num_feats){

    setCudaVariables(num_feats, num_points, X);

    //FeatureType* gradient = new FeatureType[num_feats];

    double annealed_step_size;
    const double characteristic_time = max_num_epochs * num_points / 3;
    size_t curr_num_iterations;

    for (size_t k = 0; k < max_num_epochs; k++) {
        for (size_t i = 0; i < num_points; i++) {
            FeatureType* d_x_i = &d_X[i * num_feats];
            //checkCudaErrors(cudaMemcpy(d_theta, theta, num_feats * sizeof(FeatureType), cudaMemcpyHostToDevice));
            p_gradientForSinglePoint(handle, d_theta, d_x_i, Y[i], num_feats, d_gradient);
            curr_num_iterations = k * num_points + i;
            annealed_step_size = step_size / (1.0 + (curr_num_iterations / characteristic_time));
            p_updateParameters(handle, d_theta, d_gradient, num_feats, annealed_step_size);
        }
    }
    checkCudaErrors(cudaMemcpy(theta, d_theta, num_feats * sizeof(FeatureType), cudaMemcpyDeviceToHost));
    //delete[] gradient;

    cleanUp();
}
