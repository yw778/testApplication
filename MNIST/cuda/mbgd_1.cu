#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "mbgd_1.h"
#include "mnist_utils_cuda.cuh"

/* Parallel approach to batch gradient descent using multiple "mini-batches"
rather than the whole batch. Each block is assigned to one mini-batch with
multiple threads assigned to each data point. */


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

    checkCudaErrors(cudaMalloc(&d_parameter_vector,LABEL_CLASS * num_features 
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
    checkCudaErrors(cudaMemcpy(d_parameter_vector,parameter_vector,
                                LABEL_CLASS * num_features * sizeof(FeatureType), 
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
    size_t threads_per_datapoint,
    size_t positions) {
    
    FeatureType partial_dot = 0;

    size_t thread_offset = threadIdx.x % threads_per_datapoint;

    // strided sum of element-wise products
    for (size_t j = thread_offset; j < num_features; j += threads_per_datapoint) {
        partial_dot += data_point_i[j] * parameter_vector[j];
    }

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x+positions] = partial_dot;
}


// computes gradient for a mini-batch of size batch_size in the training_set
// from starting_point
static __device__ void d_gradientForMiniBatch(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    FeatureType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t batch_size,
    size_t threads_per_datapoint,
    FeatureType* gradient){

    // array probabilities_of_each in shared_memory of size batch_size * LABEL_CLASS
    float *probabilities_of_each = (float*)&gradient[num_features * LABEL_CLASS];
    // array of groundthruth matrix
    float *probabilities_transpose = (float*)&probabilities_of_each[batch_size * LABEL_CLASS];
    // array dot_product in shared_memory of size threads_per_datapoint * batch_size * LABEL_CLASS
    float *dot_product = (float*)&probabilities_transpose[batch_size * LABEL_CLASS];
    
    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    size_t point_idx = bidx * batch_size + tidx / threads_per_datapoint;
    // thread index relative to data point
    size_t relative_tidx = threadIdx.x % threads_per_datapoint; 
    size_t point_idx_in_shmem = tidx - relative_tidx;
    size_t point_idx_in_block = tidx / threads_per_datapoint;
    // computes logistic function for each data point in the mini batch
    // size_t starting_point = point_idx * num_features;
    if (point_idx < num_data_points){
        for(size_t i = 0; i<LABEL_CLASS;i++){
            d_partialDotProduct( &data_points[point_idx * num_features], 
                                    &parameter_vector[i * num_features],
                                    dot_product, num_features, 
                                    threads_per_datapoint,
                                    i*blockDim.x);
        }
    }
    __syncthreads();


    for(size_t i=0 ; i<LABEL_CLASS ;i++){  
        for (size_t s = threads_per_datapoint / 2; s > 0; s>>=1) {
            if (relative_tidx < s) {
                dot_product[tidx+i*blockDim.x] += dot_product[tidx+s+i*blockDim.x];
            }
            __syncthreads();
        }
    }

    if (point_idx < num_data_points) {

        //calculate softmax possibility
        d_softMaxFunction1(dot_product,probabilities_of_each,
                     point_idx_in_shmem,relative_tidx,
                        point_idx_in_block, LABEL_CLASS);

        //calculate {y(i)=k}−P(y(i)=k|x(i)
        if(relative_tidx < LABEL_CLASS){
            if(labels[point_idx]==relative_tidx){
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]-=1;
                // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
            }else{                   
                // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
            }
        }
        __syncthreads();

        //debug use
        // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==0){
        //     for(size_t i=0; i<20;i++){
        //         printf("probabilities_of_each is %f\n", probabilities_of_each[i]);
        //     }   
        // } 
        // asm("trap;");  

        

        //transpose probability matrix to facilitate further computation
        d_matrixTranspose(probabilities_of_each,
                            probabilities_transpose,
                            batch_size,
                            relative_tidx,
                            point_idx_in_block);

        // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==0){
        //     for(size_t i=0; i<20;i++){
        //         printf("probabilities_transpose is %f\n", probabilities_transpose[i]);
        //     }   
        // } 
         __syncthreads();

        //Finishes computation of gradient
        size_t threads_per_mini_batch = batch_size * threads_per_datapoint;
        float factor = 1.0f/batch_size;
        // asm("trap;");
        d_matrixMatrixMultiply( data_points,
                                probabilities_transpose,
                                factor,
                                batch_size,
                                num_features,
                                threads_per_mini_batch,
                                gradient );
        // __syncthreads();

    }
}


static __global__ void p_MiniBatchGradientDescent(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    LabelType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t batch_size,
    size_t threads_per_datapoint,
    double step_size) {

    extern __shared__ FeatureType shared_memory[];
    FeatureType *gradient = shared_memory;
    size_t threads_per_mini_batch = threads_per_datapoint * batch_size;
    
    // set all gradient values to 0
    d_memset(gradient, 0, LABEL_CLASS * num_features, threads_per_mini_batch);
    __syncthreads();
    //debug use
   // if(threadIdx.x==0&&blockIdx.x==0){
   //      for(size_t i=0 ;i<PARAMETER_SIZE;i++){
   //              printf("g%f--",gradient[i]);
   //          }
   //          printf("\n\n");
   // }

    // Finds gradient for mini-batch   
    d_gradientForMiniBatch( data_points,
                            parameter_vector,
                            labels,
                            num_features,
                            num_data_points,
                            batch_size,
                            threads_per_datapoint,
                            gradient);

    __syncthreads();
    // if(threadIdx.x==0&&blockIdx.x==0){
    //     printf("after matrix muti\n");
    // }
    // asm("trap;");
    // Updates the parameters
    d_updateParameters( gradient, parameter_vector, num_features,
                        threads_per_mini_batch, step_size );
    // if(threadIdx.x==0&&blockIdx.x==0){
    //     printf("after update paramters\n");
    // }

}


void trainParallelMiniBatchGradientDescent( 
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
    
    // read configuration parameters and initialize grid and block dimensions
    double step_size = *training_options.step_size;

    const double threads_per_datapoint =
            (fieldExists(training_options.config_params, "threads_per_datapoint"))
            ? training_options.config_params["threads_per_datapoint"]
            : THREADS_PER_DATAPOINT;

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

    const dim3 block_size(threads_per_datapoint * batch_size, 1, 1);
    size_t num_blocks = DIVIDE_AND_CEIL( training_set.num_data_points,
                                            batch_size );
    const dim3 grid_size(num_blocks, 1, 1);

    FeatureType threads_per_batch = threads_per_datapoint * batch_size;
    //shared Memory for posibility, posibility transpose, shared memory and gradient
    const size_t shared_memory_size = LABEL_CLASS * batch_size * sizeof(float) 
            + LABEL_CLASS * batch_size * sizeof(float)
            + LABEL_CLASS * (threads_per_batch) * sizeof(FeatureType) 
            + LABEL_CLASS * training_set.num_features * sizeof(FeatureType);
 
    if (checkDeviceProps(shared_memory_size, block_size, grid_size)) {
        // iterate if dimensions are okay
        for (size_t k = 0; k < training_options.num_epochs; k++) {
            annealed_step_size = training_options.config_params["initial_step_size"]
                                / (1.0
                                    + (curr_num_epochs
                                       * training_set.num_data_points
                                       / characteristic_time));
            curr_num_epochs++;
            //debug use
            // printf("before enter kernal\n");
            // printf("batch size is %f, threads_per_data is %f\n",batch_size,threads_per_datapoint);
            // printf("shared memory is %d\n",shared_memory_size);

            // adjust step size with a modified version of simulated annealing

            // call kernel and check for errors
            p_MiniBatchGradientDescent
                    <<<grid_size, block_size, shared_memory_size>>>(
                            d_data_points, 
                            d_parameter_vector,
                            d_labels,
                            training_set.num_features, 
                            training_set.num_data_points, 
                            batch_size,
                            threads_per_datapoint,
                            annealed_step_size );
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaMemcpy( training_set.parameter_vector, 
                                    d_parameter_vector,
                                    LABEL_CLASS * training_set.num_features 
                                    * sizeof(FeatureType), 
                                    cudaMemcpyDeviceToHost));
    }

    *training_options.step_size = annealed_step_size; 

    cleanUp();
}
