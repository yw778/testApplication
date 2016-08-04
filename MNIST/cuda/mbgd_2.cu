# include <stdio.h>
# include <string.h>
# include <stdlib.h>
  
# include <cuda_runtime.h>
# include "mbgd_2.h"
# include "mnist_utils_cuda.cuh"

/* Parallel approach to mini batch gradient descent. In this version all
threads in the block compute for a single point before moving on to the
next one. */

// pointers to device global variables
static FeatureType *d_parameter_vector, *d_data_points;
static LabelType *d_labels;

// Allocate space for the data set, labels and parameter vector in global memory
// Then, copy values for those host variables to the device variables
static void setCudaVariables(
    size_t num_features,
    size_t num_data_points,
    FeatureType* data_points,
    LabelType* labels,
    FeatureType* parameter_vector) {

    checkCudaErrors(cudaMalloc(&d_parameter_vector, num_features 
                                * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(&d_data_points, num_data_points * num_features 
                                * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(&d_labels, num_data_points * sizeof(LabelType)));

    checkCudaErrors(cudaMemcpy(d_data_points, data_points, num_data_points
                                * num_features * sizeof(FeatureType),
                                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_labels, labels, num_data_points
                                * sizeof(LabelType), 
                                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_parameter_vector, parameter_vector,
                                num_features * sizeof(FeatureType), 
                                    cudaMemcpyHostToDevice));
}


static void cleanUp() {

    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_data_points));
    checkCudaErrors(cudaFree(d_labels));
}


static __device__ void d_partialDotProduct(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    FeatureType* shared_memory,
    size_t num_features,
    size_t threads_per_datapoint) {
    
    FeatureType partial_dot = 0;

    size_t tidx = threadIdx.x;

    // strided sum of element-wise products
    for (size_t j = tidx; j < num_features; j += threads_per_datapoint) {
        partial_dot += data_point_i[j] * parameter_vector[j];
    }

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x] = partial_dot;
}


// Finds the gradient for a minibatch. All threads in a block compute for a
// single point before moving on to the next point
static __device__ void d_gradientForMiniBatch2 (
    FeatureType* data_points,
    FeatureType* parameter_vector,
    FeatureType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t batch_size,
    size_t threads_per_mini_batch,
    FeatureType* gradient) {

    float* probabilities_of_positive = (float*)&gradient[num_features];
    float* dot_product = (float*)&probabilities_of_positive[batch_size];

    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    FeatureType* data_point_i;
    // computes logistic function for each data point in the mini batch
    for (size_t i = 0; i < batch_size; i++) {
        // index of the point with respect to the whole dataset
        size_t point_idx = bidx * batch_size + i;
        data_point_i = (FeatureType*)&data_points[point_idx * num_features];
        d_partialDotProduct( data_point_i, 
                                parameter_vector,
                                dot_product, num_features, 
                                threads_per_mini_batch );
        
        __syncthreads();

        // sum reduce to find dot product
        for (size_t s = threads_per_mini_batch / 2; s > 0; s>>=1) {
            if (tidx < s){
                dot_product[tidx] += dot_product[tidx + s];
            }
        }
       
        __syncthreads();

        probabilities_of_positive[i] = d_logisticFunction(*dot_product)
                 - labels[bidx * batch_size + i];

        __syncthreads();
    }
    
    float factor = 1.0f / batch_size;
    // finish computation of gradient
    d_matrixVectorMultiply( data_points,
                            probabilities_of_positive,
                            factor,
                            batch_size,
                            num_features,
                            threads_per_mini_batch,
                            gradient );  
}


static __global__ void p_MiniBatchGradientDescent2(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    LabelType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t batch_size,
    size_t threads_per_mini_batch,
    double step_size) {

    extern __shared__ FeatureType shared_memory[];
    FeatureType *gradient = shared_memory;

    d_memset(gradient, 0, num_features, threads_per_mini_batch); 

    // Finds gradient for mini-batch
    d_gradientForMiniBatch2( data_points,
                            parameter_vector,
                            labels,
                            num_features,
                            num_data_points,
                            batch_size,
                            threads_per_mini_batch,
                            gradient );

    __syncthreads();

    // Updates the parameters
    d_updateParameters( gradient, parameter_vector, num_features,
                        threads_per_mini_batch, step_size );
}


void trainParallelMiniBatchGradientDescent2( 
    DataSet training_set,
    TrainingOptions training_options ) {

    // shuffle data points
    /* shuffleKeyValue( training_set.data_points, training_set.labels, 
                     training_set.num_data_points, training_set.num_features ); */

    setCudaVariables( training_set.num_features,
                      training_set.num_data_points,
                      training_set.data_points, 
                      training_set.labels, 
                      training_set.parameter_vector );

    double step_size = *training_options.step_size;

    const double threads_per_mini_batch =
            (fieldExists(training_options.config_params, "threads_per_mini_batch"))
            ? training_options.config_params["threads_per_mini_batch"]
            : THREADS_PER_MINI_BATCH;

    const double batch_size =
            (fieldExists(training_options.config_params, "batch_size"))
            ? training_options.config_params["batch_size"]
            : BATCH_SIZE;

    const double characteristic_time =
            (fieldExists(training_options.config_params, "characteristic_time"))
            ? training_options.config_params["characteristic_time"]
            : CHARACTERISTIC_TIME;

    size_t curr_num_epochs =
            (fieldExists(training_options.config_params, "curr_num_epochs"))
            ? training_options.config_params["curr_num_epochs"]
            : 0;

    double annealed_step_size = step_size;

    const dim3 block_size(threads_per_mini_batch, 1, 1);
    size_t num_blocks = DIVIDE_AND_CEIL( training_set.num_data_points,
                                            batch_size );
    const dim3 grid_size(num_blocks, 1, 1);

    const size_t shared_memory_size = batch_size * sizeof(float) 
            + (threads_per_mini_batch) 
            * sizeof(FeatureType) + training_set.num_features
            * sizeof(FeatureType);
 
    if (checkDeviceProps(shared_memory_size, block_size, grid_size)) {
        // iterate if dimensions are okay
        for (size_t k = 0; k < training_options.num_epochs; k++) {

            // adjust step size with a modified version of simulated annealing
            annealed_step_size = training_options.config_params["initial_step_size"]
                                / (1.0
                                    + (curr_num_epochs
                                       * training_set.num_data_points
                                       / characteristic_time));
            curr_num_epochs++;

            // call kernel and check for errors
            p_MiniBatchGradientDescent2
                    <<<grid_size, block_size, shared_memory_size>>>(
                        d_data_points, 
                            d_parameter_vector,
                            d_labels,
                            training_set.num_features, 
                            training_set.num_data_points, 
                            batch_size,
                            threads_per_mini_batch,
                            annealed_step_size );
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaMemcpy( training_set.parameter_vector, 
                                    d_parameter_vector,
                                    training_set.num_features 
                                    * sizeof(FeatureType), 
                                    cudaMemcpyDeviceToHost));
    }

    *training_options.step_size = annealed_step_size; 

    cleanUp();
}

