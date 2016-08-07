#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "sgd_single_point.h"
#include "mnist_utils_cuda.cuh"


/*
 * Parallel approach to Stochastic Gradient Descent #2 - CUDA:
 * Parallel computation of gradient and synchronous update of the parameter
 * vector with atomic operations. Several threads work on every data point, and
 * several datapoints are processed concurrently.
 *
 * Note: This version only works when dimensions of variables are powers of 2
 */


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

    checkCudaErrors(cudaMalloc(
        &d_parameter_vector, LABEL_CLASS * num_features * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(
        &d_data_points, num_data_points * num_features * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(
        &d_labels, num_data_points * sizeof(LabelType)));

    checkCudaErrors(cudaMemcpy(
        d_data_points,
        data_points,
        num_data_points * num_features * sizeof(FeatureType),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_labels,
        labels,
        num_data_points * sizeof(LabelType),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_parameter_vector,
        parameter_vector,
        LABEL_CLASS * num_features * sizeof(FeatureType),
        cudaMemcpyHostToDevice));
}

// Free space in global memory
static void cleanUp() {
    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_data_points));
    checkCudaErrors(cudaFree(d_labels));
}

// Computes a fraction of the dot product when N threads are working on a
// single data point. The elements processed by each thread are those
// separated by a stride equal to N with an offset given by the thread index % N
static __device__ void d_partialDotProduct(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    FeatureType* shared_memory,
    size_t num_features,
    size_t threads_per_datapoint,
    size_t position) {
    //memset to 0
    FeatureType partial_dot = 0;

    size_t thread_offset = threadIdx.x % threads_per_datapoint;

    // strided sum of element-wise products
    for (size_t j = thread_offset; j < num_features; j+=threads_per_datapoint)
        partial_dot += data_point_i[j] * parameter_vector[j];

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x + position] = partial_dot;
}

// updates parameter vector in parallel when N threads are working on each point
static __device__ void d_updateParameters(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    size_t num_features,
    size_t threads_per_datapoint,
    size_t point_idx_in_block,
    size_t relative_tidx,
    FeatureType* step_size_times_prob_i_minus_label_i) {

    // printf("enter update parameters in sgd_single_point\n");

    // size_t thread_offset = threadIdx.x % threads_per_datapoint;

    // finishes computation of gradient and updates shared parameter_vector

    //debug use

    // if(relative_tidx==0&&blockIdx.x==0){
    //     for(size_t i=0; i<LABEL_CLASS;i++){
    //         printf("gradient-%f--\n", step_size_times_prob_i_minus_label_i[i]);
    //     }   
    // } 
    // asm("trap;"); 

     // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1){
     //        for(size_t i=0; i<num_features;i++){
     //            printf("p+%f--", parameter_vector[i]);
     //        }   

     //        printf("\n\n\n\n\n\n\n\n");
     //    } 
     //    asm("trap;");

    for(size_t i=0;i<LABEL_CLASS;i++){
        for (size_t j = relative_tidx; j < num_features; j += threads_per_datapoint){

            // the gradient is: x * (pi - y)
            FeatureType gradient_times_step_size = data_point_i[j] 
                * step_size_times_prob_i_minus_label_i[point_idx_in_block * LABEL_CLASS+i];
               // // //debug use
            if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1&&i==6){
                printf("gradient_times_step_size is  %f\n",gradient_times_step_size);
                printf("data_point_i[j] is  %f\n",data_point_i[j]);
                printf("minus is  %f\n",step_size_times_prob_i_minus_label_i[point_idx_in_block * LABEL_CLASS+i]);
                printf("+before add is %d -- %f\n",(j+i * num_features),parameter_vector[j+i * num_features]);
            }
            // asm("trap;"); 
            // if(relative_tidx==0&&blockIdx.x==0){
            //     printf("before add is %d %f\n",j+i * num_features, parameter_vector[j+i * num_features]);
            // } 
            // asm("trap;");
            atomicAdd(&parameter_vector[j+i * num_features], - gradient_times_step_size);
            // // //debug use
            // if(relative_tidx==0&&blockIdx.x==0){
            //     printf("gradient is  %f\n",parameter_vector[j+i* num_features]);
            // }
            if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1&&i==6){
                printf("-after add %d -- %f\n",(j+i * num_features),parameter_vector[j+i * num_features]);
            } 
        }
        // if(i==8)
        // asm("trap;"); 
    }
    asm("trap;"); 

     // debug use
    // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1){
    //     for(size_t i=0; i<num_features;i++){
    //         printf("p-%f--", parameter_vector[i]);
    //     }
    //     printf("\n\n\n");   
    // } 
    // asm("trap;"); 
}

// Kernel for Parallel Stochastic Gradient Descent in CUDA using
// shared parameter vector
static __global__ void p_SgdWithSharedParameterVector(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    LabelType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t threads_per_datapoint,
    double step_size) {

    extern __shared__ FeatureType shared_memory[];
    float *probabilities_of_each = (float*)&shared_memory[blockDim.x 
            * LABEL_CLASS];
    // printf("in the kernal \n");


    // computes several indexes, offsets and strides to simplify further code
    size_t tidx = threadIdx.x;
    size_t points_per_block = (blockDim.x / threads_per_datapoint);
    size_t point_idx = (blockIdx.x * points_per_block)
                     + (tidx / threads_per_datapoint);
    // index relative to the datapoint instead of the block
    size_t relative_tidx = tidx % threads_per_datapoint;
    size_t point_idx_in_shmem = tidx - relative_tidx;
    size_t point_idx_in_block = tidx / threads_per_datapoint;

    // float *probabilities_of_sum = (float*)&probabilities_of_each[LABEL_CLASS 
        // * points_per_block];

    FeatureType* data_point_i = NULL;

    // make sure the threads don't go out of bounds
    if (point_idx < num_data_points) {

        data_point_i = (FeatureType*) &data_points[point_idx * num_features];

        // compute partial dot product
        for(size_t i = 0; i<LABEL_CLASS;i++){
            
            d_partialDotProduct(
                data_point_i,
                &parameter_vector[i * num_features],
                shared_memory,
                num_features,
                threads_per_datapoint,
                i*blockDim.x);
        }
    }
    //debug use
    // for(size_t i=blockDim/2; i<blockDim.x;i++){
    //     printf("shared memory is %f\n", shared_memory[i]);
    // }   

    // printf("block size is %d\n",blockDim.x);


    __syncthreads();

    // sum-reduce the results of partial dot product to get final result
    for(size_t i=0 ; i<LABEL_CLASS ;i++){    
        for (size_t s = threads_per_datapoint / 2; s > 0; s>>=1) {
            if (relative_tidx < s) {
                shared_memory[tidx+i*blockDim.x] += shared_memory[tidx+s+i*blockDim.x];
            }
            __syncthreads();
        }
    }
    //__syncthreads();
    // make sure the threads don't go out of bounds
    if (point_idx < num_data_points) {
        // double probability_of_positive =
        //     d_softMaxFunction(shared_memory[point_idx_in_shmem]);
        d_softMaxFunction(shared_memory,probabilities_of_each,
                 point_idx_in_shmem,relative_tidx,
                    point_idx_in_block, LABEL_CLASS);
        //debug use
        // if(relative_tidx==0&&blockIdx.x==0){
        //     for(size_t i=10; i<20;i++){
        //         printf("shared memory is %f\n", probabilities_of_each[i]);
        //     }   
        // } 
        // asm("trap;");  
        //calculate step_size_times_prob_i_minus_label_i, store in the same position
        if(relative_tidx < LABEL_CLASS){
            if(labels[point_idx]==relative_tidx){
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]-=1;
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
            }else{                   
                // if(relative_tidx==0&&blockIdx.x==0){
                //     printf("before inupdating..%f\n",probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]);
                //      printf("step size is %f\n",step_size);
                // }

                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
                // if(relative_tidx==0&&blockIdx.x==0){
                //     printf("inupdating..%f\n",probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]);
                // }
            }
        }
        //debug use
        // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1){
        //     for(size_t i=10; i<20;i++){
        //         printf("after parameter is factored %f\n", probabilities_of_each[i]);
        //     }   
        //     printf("label is %f\n",labels[point_idx]);
        // } 
        // asm("trap;");  

        // double step_size_times_prob_i_minus_label_i =
        //     (probability_of_positive - labels[point_idx]) * step_size;
        __syncthreads();
        // debug use
        //    if(relative_tidx==0&&blockIdx.x==0){
        //     // printf("blockIdx.x is %d--tid is %d\n",blockIdx.x,threadIdx.x);
        //     // printf("point_index is %d----%f\n",point_idx,labels[point_idx]);
        //     for(size_t i=0; i<10;i++){
        //         printf("after parameter is factored %f\n", probabilities_of_each[i]);
        //     }   
        // } 
        // asm("trap;");

        // debug use
        // printf("before update parameters \n");
        // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==1){
        //     for(size_t i=0; i<num_features;i++){
        //         printf("p+%f--", parameter_vector[i]);
        //     }   

        //     printf("\n\n\n\n\n\n\n\n");
        // } 
        // asm("trap;");

        d_updateParameters(
            data_point_i,
            parameter_vector,
            num_features,
            threads_per_datapoint,
            point_idx_in_block,
            relative_tidx,
            probabilities_of_each);
        // debug use
        // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block ==1){
        //     for(size_t i=0; i<PARAMETER_SIZE;i++){
        //         printf("p-%f--\n", parameter_vector[i]);
        //     }   
        // } 
    }   

}


// Executes serial implementation of stochastic gradient descent for logistic
// regression with a fixed number of iterations
// config_params: {step_size, threads_per_datapoint, datapoints_per_block}
void trainParallelStochasticGradientDescent2(
    DataSet training_set,
    TrainingOptions training_options) {

    // shuffle datapoints in order to add more stochasticity
    // shuffleKeyValue(
    //     training_set.data_points,
    //     training_set.labels,
    //     training_set.num_data_points,
    //     training_set.num_features);

    // allocate device memory
    setCudaVariables(
        training_set.num_features,
        training_set.num_data_points,
        training_set.data_points,
        training_set.labels,
        training_set.parameter_vector);


    // read configuration parameters and initialize grid and block dimensions
    double step_size = *training_options.step_size;

    const double threads_per_datapoint =
            (fieldExists(training_options.config_params, "threads_per_datapoint"))
            ? training_options.config_params["threads_per_datapoint"]
            : THREADS_PER_DATAPOINT;

    const double datapoints_per_block =
            (fieldExists(training_options.config_params, "datapoints_per_block"))
            ? training_options.config_params["datapoints_per_block"]
            : DATAPOINTS_PER_BLOCK;

    const double characteristic_time =
            (fieldExists(training_options.config_params, "characteristic_time"))
            ? training_options.config_params["characteristic_time"]
            : CHARACTERISTIC_TIME;

    size_t curr_num_epochs =
            (fieldExists(training_options.config_params, "curr_num_epochs"))
            ? training_options.config_params["curr_num_epochs"]
            : 0;

    double annealed_step_size = step_size;

    const dim3 block_size(
        threads_per_datapoint * datapoints_per_block,
        1,
        1);
    const dim3 grid_size(
        DIVIDE_AND_CEIL(training_set.num_data_points, datapoints_per_block),
        1,
        1);

    const size_t shared_memory_size = block_size.x * sizeof(FeatureType) * LABEL_CLASS
        + datapoints_per_block * sizeof(FeatureType) * LABEL_CLASS ;
        // + datapoints_per_block * sizeof(FeatureType);

    // printf("memosize is %d",shared_memory_size);
    // exit(1);

    // check that the resulting grid and block dimensions
    // dont' violate device limits
    if(checkDeviceProps(shared_memory_size, block_size, grid_size)) {

        // iterate if dimensions are okay
        for (size_t k = 0; k < training_options.num_epochs; k++) {

            // adjust step size with a modified version of simulated annealing
            annealed_step_size = training_options.config_params["initial_step_size"]
                                 / (1.0
                                    + (curr_num_epochs
                                       * training_set.num_data_points
                                       / characteristic_time));
            curr_num_epochs++;
            // printf("before kernal is launched\n");

            // for(size_t i=0 ;i<PARAMETER_SIZE;i++){
            //     printf("p%f--",training_set.parameter_vector[i]);
            // }
            // printf("\n\n");

            // call kernel and check for errors
            p_SgdWithSharedParameterVector
                <<<grid_size, block_size, shared_memory_size>>>(
                    d_data_points,
                    d_parameter_vector,
                    d_labels,
                    training_set.num_features,
                    training_set.num_data_points,
                    threads_per_datapoint,
                    annealed_step_size);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());

        }
        // copy results from device memory back to host
        checkCudaErrors(cudaMemcpy(
            training_set.parameter_vector,
            d_parameter_vector,
            LABEL_CLASS * training_set.num_features * sizeof(FeatureType),
            cudaMemcpyDeviceToHost));

    }

    *training_options.step_size = annealed_step_size;

    cleanUp();
}
