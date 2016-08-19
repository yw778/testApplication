# include <stdio.h>
# include <string.h>
# include <stdlib.h>
  
# include <cuda_runtime.h>
# include "mbgd_2_2.h"
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
    checkCudaErrors(cudaMemcpy(d_parameter_vector, parameter_vector,
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

    // size_t thread_offset = threadIdx.x % threads_per_datapoint;
    size_t tidx = threadIdx.x;

    // strided sum of element-wise products
    for (size_t j = tidx; j < num_features; j += threads_per_datapoint) {
        partial_dot += data_point_i[j] * parameter_vector[j];
    }

    // result of the partial dot product is stored in shared memory
    shared_memory[tidx+positions] = partial_dot;
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

    size_t tidx = threadIdx.x;
    // size_t thread_offset = threadIdx.x % threads_per_datapoint;
    size_t num_thread_each_label = threads_per_datapoint / LABEL_CLASS;
    //index relative to each label(corresponding to 784 parameter) 
    //Eg: 320 thread for 10 label -> each label 32 thread
    size_t tidx_label =  tidx / num_thread_each_label;
    size_t relative_tidx_label =  tidx % num_thread_each_label;
    // strided sum of element-wise products concurrently in 10 dimentions
    for (size_t j = relative_tidx_label; j < num_features; j+= num_thread_each_label)
        partial_dot += data_point_i[j] * parameter_vector[j + tidx_label * num_features];

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

    // float* probabilities_of_positive = (float*)&gradient[num_features];
    // float* dot_product = (float*)&probabilities_of_positive[batch_size];

     // array probabilities_of_each in shared_memory of size batch_size * LABEL_CLASS
    float *probabilities_of_each = (float*)&gradient[num_features * LABEL_CLASS];
    // array of transpose of probabilities matrix
    // Eg: [1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10] -> [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10] 
    float *probabilities_transpose = (float*)&probabilities_of_each[batch_size * LABEL_CLASS];
    // array dot_product in shared_memory of size threads_per_datapoint * batch_size
    float *dot_product = (float*)&probabilities_transpose[batch_size * LABEL_CLASS];

    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    // relative_idx of the whole batch
    // size_t relative_tidx = threadIdx.x % threads_per_mini_batch; 
    // variables used to calculate matrix transpose
    // size_t threads_per_datapoint = threads_per_mini_batch / batch_size;
    // size_t relative_tidx = tidx % threads_per_datapoint; 
    // size_t point_idx_in_block = tidx / threads_per_datapoint;
    // calculate index relative to each label
    size_t num_thread_each_label = threads_per_mini_batch / LABEL_CLASS;
    size_t tidx_label =  tidx / num_thread_each_label;
    size_t relative_tidx_label = tidx % num_thread_each_label;

    FeatureType* data_point_i;
    // computes softmax function for each data point in the mini batch
    for (size_t j = 0; j < batch_size; j++) {
        // index of the point with respect to the whole dataset
        size_t point_idx = bidx * batch_size + j;
        data_point_i = (FeatureType*)&data_points[point_idx * num_features];
        // d_partialDotProduct( data_point_i, 
        //                         parameter_vector,
        //                         dot_product, num_features, 
        //                         threads_per_mini_batch );
        if (point_idx < num_data_points){
            // for(size_t i = 0; i<LABEL_CLASS; i++){
            //     d_partialDotProduct( data_point_i,
            //                             &parameter_vector[i * num_features],
            //                             dot_product, num_features, 
            //                             threads_per_mini_batch,
            //                             i*blockDim.x);
            // }
            d_partialMatrixVectorProduct(
                data_point_i,
                parameter_vector,
                dot_product,
                num_features,
                threads_per_mini_batch);
        }

        __syncthreads();

        // sum reduce to find dot product
        // for (size_t s = threads_per_mini_batch / 2; s > 0; s>>=1) {
        //     if (tidx < s){
        //         dot_product[tidx] += dot_product[tidx + s];
        //     }
        // }
        //  __syncthreads();

        // for(size_t i=0 ; i<LABEL_CLASS ;i++){  
        for (size_t s = num_thread_each_label / 2; s > 0; s>>=1) {
            if (relative_tidx_label < s) {
                dot_product[tidx] += dot_product[tidx+s];
            }
            __syncthreads();
        }
        // }
       
        // d_softMaxFunction3(dot_product, probabilities_of_each,
        //               tidx, j, LABEL_CLASS);
        if (point_idx < num_data_points) {
            d_softMaxFunction4(dot_product, probabilities_of_each,
                          tidx, j, num_thread_each_label);


            if(tidx < LABEL_CLASS){
                if(labels[point_idx]==tidx){
                    probabilities_of_each[j * LABEL_CLASS+tidx]-=1;
                    // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
                }else{                   
                    // probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx]*=step_size;
                }
            }
            __syncthreads();
        }    

        // probabilities_of_positive[i] = d_logisticFunction(*dot_product)
        //          - labels[bidx * batch_size + i];

    }
    // use the first LABEL_CLASS * batch_size thread to update
    // the probability matrix
    // the result is p=p-groundtruth
    // if(threadIdx.x < (LABEL_CLASS * batch_size)){
    //     //calcuate the first  LABEL_CLASS * batch_size thread
    //     // idx relative to the label
    //     size_t tidx_class = threadIdx.x / LABEL_CLASS;
    //     size_t relative_tidx_class = threadIdx.x % LABEL_CLASS;

    //     if(labels[point_idx]==relative_tidx_class){
    //             probabilities_of_each[tidx_class * LABEL_CLASS+relative_tidx_class]-=1;
    //         }   

    // }
    //debug use
    // if(tidx==0&&blockIdx.x==0){
    //         for (int i = 0; i < 20; ++i)
    //         {
    //             printf("p is %f\n",probabilities_of_each[i]);
    //         }
    //         printf("\n\n");
    // } 
   // asm("trap;");



    

    // d_matrixTranspose(probabilities_of_each,
    //                         probabilities_transpose,
    //                         batch_size,
    //                         relative_tidx,
    //                         point_idx_in_block);

    d_matrixTranspose2(probabilities_of_each,
                            probabilities_transpose,
                            batch_size);

    __syncthreads();


    // if(tidx==0&&blockIdx.x==0){
    //         for (int i = 0; i < 21; ++i)
    //         {
    //             printf(" after p is %f\n",probabilities_transpose[i]);
    //         }
    // } 
    // asm("trap;");
    float factor = 1.0f / batch_size;
    // finish computation of gradient
    // d_matrixVectorMultiply( data_points,
    //                         probabilities_of_positive,
    //                         factor,
    //                         batch_size,
    //                         num_features,
    //                         threads_per_mini_batch,
    //                         gradient );  

    // d_matrixMatrixMultiply( data_points,
    //                             probabilities_transpose,
    //                             factor,
    //                             batch_size,
    //                             num_features,
    //                             threads_per_mini_batch,
    //                             gradient );
    d_matrixMatrixMultiply2( data_points,
                                probabilities_transpose,
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

    d_memset(gradient, 0, LABEL_CLASS * num_features, threads_per_mini_batch); 

    __syncthreads();
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

    // if(threadIdx.x==0&&blockIdx.x==0){
    //         for (int i = 0; i < PARAMETER_SIZE; ++i)
    //         {
    //             printf(" parameters is %f\n",gradient[i]);
    //         }
    // } 
    // asm("trap;");

    // Updates the parameters
    d_updateParameters( gradient, parameter_vector, num_features,
                        threads_per_mini_batch, step_size );

    // if(threadIdx.x==0&&blockIdx.x==0){
    //         for (int i = 0; i < PARAMETER_SIZE; ++i)
    //         {
    //             printf(" parameters is %f\n",parameter_vector[i]);
    //         }
    // } 
    // asm("trap;");
}


void trainParallelMiniBatchGradientDescent22( 
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

      //shared Memory for posibility, posibility transpose, dot product and gradient
    const size_t shared_memory_size = LABEL_CLASS * batch_size * sizeof(float) 
            + LABEL_CLASS * batch_size * sizeof(float)
            + (threads_per_mini_batch) * sizeof(FeatureType) 
            + LABEL_CLASS * training_set.num_features * sizeof(FeatureType);
 
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
                                    LABEL_CLASS * training_set.num_features 
                                    * sizeof(FeatureType), 
                                    cudaMemcpyDeviceToHost));
    }

    *training_options.step_size = annealed_step_size; 

    cleanUp();
}

