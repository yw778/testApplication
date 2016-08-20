#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "mbgd_1_2.h"
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


// two - dimentional parallelism dot product
// work for threads that are multiple of both 10 and 32
static __device__ void d_partialMatrixVectorProduct(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    FeatureType* shared_memory,
    size_t num_features,
    size_t threads_per_datapoint) {
    //memset to 0
    FeatureType partial_dot = 0;

    size_t thread_offset = threadIdx.x % threads_per_datapoint;
    size_t num_thread_each_label = threads_per_datapoint / LABEL_CLASS;
    //index relative to each label(corresponding to 784 parameter) 
    //Eg: 320 thread for 10 label -> each label 32 thread
    size_t tidx_label =  thread_offset / num_thread_each_label;
    size_t relative_tidx_label =  thread_offset % num_thread_each_label;
    // strided sum of element-wise products concurrently in 10 dimentions
    for (size_t j = relative_tidx_label; j < num_features; j+= num_thread_each_label)
        partial_dot += data_point_i[j] * parameter_vector[j + tidx_label * num_features];

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x] = partial_dot;
}


// computes gradient for a mini-batch of size batch_size in the training_set
// from starting_point
// static __device__ void d_gradientForMiniBatch(
//     FeatureType* data_points,
//     FeatureType* parameter_vector,
//     FeatureType* labels,
//     size_t num_features,
//     size_t num_data_points,
//     size_t batch_size,
//     size_t threads_per_datapoint,
//     FeatureType* gradient){

//     // array probabilities_of_each in shared_memory of size batch_size * LABEL_CLASS
//     float *probabilities_of_each = (float*)&gradient[num_features * LABEL_CLASS];
//     // array of transpose of probabilities matrix
//     //Eg: [1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10] -> [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10] 
//     float *probabilities_transpose = (float*)&probabilities_of_each[batch_size * LABEL_CLASS];
//     // array dot_product in shared_memory of size threads_per_datapoint * batch_size
//     float *dot_product = (float*)&probabilities_transpose[batch_size * LABEL_CLASS];
    
//     size_t tidx = threadIdx.x;
//     size_t bidx = blockIdx.x;
//     size_t point_idx = bidx * batch_size + tidx / threads_per_datapoint;
//     // thread index relative to data point
//     size_t relative_tidx = threadIdx.x % threads_per_datapoint; 
//     size_t point_idx_in_shmem = tidx - relative_tidx;
//     size_t point_idx_in_block = tidx / threads_per_datapoint;

//     //index relative to each label(corresponding to 784 parameter) 
//     //Eg: 320 thread for 10 label -> each label 32 thread
//     size_t num_thread_each_label = threads_per_datapoint / LABEL_CLASS;
//     // size_t tidx_label =  relative_tidx / num_thread_each_label;
//     size_t relative_tidx_label =  relative_tidx % num_thread_each_label;

//     // computes softmax function for each data point in the mini batch
//     // size_t starting_point = point_idx * num_features;
//     if (point_idx < num_data_points){
//         // for(size_t i = 0; i<LABEL_CLASS;i++){
//         // d_partialDotProduct( &data_points[point_idx * num_features], 
//         //                         &parameter_vector[i * num_features],
//         //                         dot_product, num_features, 
//         //                         threads_per_datapoint,
//         //                         i*blockDim.x);
//         // }
//         d_partialMatrixVectorProduct(
//                 &data_points[point_idx * num_features], 
//                 parameter_vector,
//                 dot_product,
//                 num_features,
//                 threads_per_datapoint);
        
//     }
//     __syncthreads();


//     // sum-reduce the results of partial matrix-vector product to get final result
//     // for(size_t i=0 ; i<LABEL_CLASS ;i++){  
//     for (size_t s = num_thread_each_label / 2; s > 0; s>>=1) {
//         if (relative_tidx_label < s) {
//             dot_product[tidx] += dot_product[tidx+s];
//         }
//         __syncthreads();
//     }
//     // }

//     if (point_idx < num_data_points) {

//         //calculate softmax possibility
//         // d_softMaxFunction1(dot_product,probabilities_of_each,
//         //              point_idx_in_shmem,relative_tidx,
//         //                 point_idx_in_block, LABEL_CLASS);
//         d_softMaxFunction2(dot_product,probabilities_of_each,
//                 point_idx_in_shmem,relative_tidx, point_idx_in_block,
//                 num_thread_each_label);

//         //calculate {y(i)=k}−P(y(i)=k|x(i)
//         if(relative_tidx < LABEL_CLASS){
//             if(labels[point_idx]==relative_tidx){
//                 probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]-=1;
//                 // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
//             }else{                   
//                 // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
//             }
//         }
//         __syncthreads();

//         //debug use
//         // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==0){
//         //     for(size_t i=0; i<20;i++){
//         //         printf("probabilities_of_each is %f\n", probabilities_of_each[i]);
//         //     }   
//         // } 
//         // asm("trap;");  

        

//         //transpose probability matrix to facilitate further computation
//         d_matrixTranspose(probabilities_of_each,
//                             probabilities_transpose,
//                             batch_size,
//                             relative_tidx,
//                             point_idx_in_block);

//         // if(relative_tidx==0&&blockIdx.x==0&&point_idx_in_block==0){
//         //     for(size_t i=0; i<20;i++){
//         //         printf("probabilities_transpose is %f\n", probabilities_transpose[i]);
//         //     }   
//         // } 
//          __syncthreads();

//         //Finishes computation of gradient
//         size_t threads_per_mini_batch = batch_size * threads_per_datapoint;
//         float factor = 1.0f/batch_size;
//         d_matrixMatrixMultiply2( data_points,
//                                 probabilities_transpose,
//                                 factor,
//                                 batch_size,
//                                 num_features,
//                                 threads_per_mini_batch,
//                                 gradient );
//         // __syncthreads();

//     }
// }


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
    // FeatureType *gradient = shared_memory;
    size_t threads_per_mini_batch = threads_per_datapoint * batch_size;
    

    float *dot_product = shared_memory;
    // array probabilities_of_each in shared_memory of size batch_size * LABEL_CLASS
    float *probabilities_of_each = (float*)&dot_product[threads_per_mini_batch];
    // array of transpose of probabilities matrix
    //Eg: [1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10] -> [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10] 
    // float *probabilities_transpose = (float*)&probabilities_of_each[batch_size * LABEL_CLASS];
    // array dot_product in shared_memory of size threads_per_datapoint * batch_size
    // float *dot_product = (float*)&probabilities_transpose[batch_size * LABEL_CLASS];
    
    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    size_t point_idx = bidx * batch_size + tidx / threads_per_datapoint;
    // thread index relative to data point
    size_t relative_tidx = threadIdx.x % threads_per_datapoint; 
    size_t point_idx_in_shmem = tidx - relative_tidx;
    size_t point_idx_in_block = tidx / threads_per_datapoint;

    //index relative to each label(corresponding to 784 parameter) 
    //Eg: 320 thread for 10 label -> each label 32 thread
    size_t num_thread_each_label = threads_per_datapoint / LABEL_CLASS;
    // size_t tidx_label =  relative_tidx / num_thread_each_label;
    size_t relative_tidx_label =  relative_tidx % num_thread_each_label;

    // computes softmax function for each data point in the mini batch
    // size_t starting_point = point_idx * num_features;
    if (point_idx < num_data_points){
        
         d_partialMatrixVectorProduct(
                &data_points[point_idx * num_features], 
                parameter_vector,
                dot_product,
                num_features,
                threads_per_datapoint);
        
    }
    __syncthreads();


    // sum-reduce the results of partial matrix-vector product to get final result

    for (size_t s = num_thread_each_label / 2; s > 0; s>>=1) {
        if (relative_tidx_label < s) {
            dot_product[tidx] += dot_product[tidx+s];
        }
        __syncthreads();
    }

    if (point_idx < num_data_points) {


        d_softMaxFunction2(dot_product, probabilities_of_each,
                point_idx_in_shmem, relative_tidx, point_idx_in_block,
                num_thread_each_label);

        float reduced_stepsize = step_size / batch_size;

        //calculate {y(i)=k}−P(y(i)=k|x(i)
        if(relative_tidx < LABEL_CLASS){
            if(labels[point_idx]==relative_tidx){
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] -= 1;
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= reduced_stepsize;
            }else{                   
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= reduced_stepsize;
            }
        }
        __syncthreads();


        d_updateParameters1( data_points, probabilities_of_each, parameter_vector, num_features,
                            batch_size, threads_per_mini_batch, reduced_stepsize);
    

    }

}


void trainParallelMiniBatchGradientDescent12( 
    DataSet training_set,
    TrainingOptions training_options ) {

    // shuffle data points
    /* shuffleKeyValue( training_set.data_points, training_set.labels, 
                     training_set.num_data_points, training_set.num_features ); */

    setCudaVariables( training_set.num_features,
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
    //shared Memory for posibility, posibility transpose, dot product and gradient
    const size_t shared_memory_size = LABEL_CLASS * batch_size * sizeof(float) 
            + threads_per_batch * sizeof(FeatureType);
            // + LABEL_CLASS * batch_size * sizeof(float)
           
            // + LABEL_CLASS * training_set.num_features * sizeof(FeatureType);

 
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
