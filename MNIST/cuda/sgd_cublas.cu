#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cublas_v2.h>

#include "sgd_cublas.h"
#include "mnist_utils_cuda.cuh"


static FeatureType *d_parameter_vector, *d_data_points, *d_gradient;
static cublasHandle_t handle;

static void setCudaVariables(
    size_t num_features,
    size_t num_data_points,
    FeatureType* data_points) {

    checkCuBlasErrors(cublasCreate(&handle));

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    const size_t size_of_datapoint = num_features * sizeof(FeatureType);

    checkCudaErrors(cudaMalloc(&d_parameter_vector, size_of_datapoint));
    checkCudaErrors(cudaMalloc(&d_gradient, size_of_datapoint));
    checkCudaErrors(cudaMalloc(&d_data_points, (size_of_datapoint
                                                * num_data_points)));

    checkCudaErrors(cudaMemcpy(d_data_points,
                               data_points,
                               (size_of_datapoint * num_data_points),
                               cudaMemcpyHostToDevice));
}

static void cleanUp() {

    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_gradient));
    checkCudaErrors(cudaFree(d_data_points));

    cublasDestroy(handle);
}

// computes gradient for a single datapoint using cublas
static void p_gradientForSinglePoint (
    cublasHandle_t handle,
    FeatureType* d_parameter_vector,
    FeatureType* d_data_point_i,
    LabelType label,
    size_t num_features,
    FeatureType* d_gradient) {

    float probability_of_positive = p_logisticFunction(
        handle,
        d_parameter_vector,
        d_data_point_i,
        num_features);

    checkCudaErrors(
        cudaMemset(d_gradient, 0, num_features * sizeof(FeatureType)));

    p_add_vectors(
        handle,
        d_gradient,
        d_data_point_i,
        num_features,
        (probability_of_positive - label));
}

// executes serial implementation of stochastic gradient descent
// for logistic regression with a fixed number of iterations
// config_params: {step_size}
void trainStochasticGradientDescent1(
    DataSet training_set,
    TrainingOptions training_options){

    setCudaVariables(
        training_set.num_features,
        training_set.num_data_points,
        training_set.data_points);

    //FeatureType* gradient = new FeatureType[training_set.num_features];

    // read configuration parameters
    double step_size = *training_options.step_size;

    const double characteristic_time =
            (fieldExists(training_options.config_params, "characteristic_time"))
            ? training_options.config_params["characteristic_time"]
            : CHARACTERISTIC_TIME;

    size_t curr_num_epochs =
            (fieldExists(training_options.config_params, "curr_num_epochs"))
            ? training_options.config_params["curr_num_epochs"]
            : 0;

    double annealed_step_size = step_size;

    for (size_t k = 0; k < training_options.num_epochs; k++) {

        // simulated annealing (reduces step size as it converges)
        annealed_step_size = training_options.config_params["initial_step_size"]
                             / (1.0
                                + (curr_num_epochs
                                   * training_set.num_data_points
                                   / characteristic_time));
        curr_num_epochs++;

        for (size_t i = 0; i < training_set.num_data_points; i++) {
            FeatureType* d_data_point_i = &d_data_points[i * training_set.num_features];
            //checkCudaErrors(cudaMemcpy(d_parameter_vector, training_set.parameter_vector, training_set.num_features * sizeof(FeatureType), cudaMemcpyHostToDevice));
            p_gradientForSinglePoint(handle, d_parameter_vector, d_data_point_i, training_set.labels[i], training_set.num_features, d_gradient);
            p_updateParameters(handle, d_parameter_vector, d_gradient, training_set.num_features, annealed_step_size);
        }
    }
    checkCudaErrors(cudaMemcpy(training_set.parameter_vector, d_parameter_vector, (training_set.num_features * sizeof(FeatureType)), cudaMemcpyDeviceToHost));
    //delete[] gradient;

    *training_options.step_size = annealed_step_size;

    cleanUp();
}
