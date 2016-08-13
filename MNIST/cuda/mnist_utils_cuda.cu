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

// Grabdient = probility_matrix_transpose * datapoint_matrix
__device__ void d_matrixMatrixMultiply(
    FeatureType* datapoint_matrix,
    FeatureType* probility_matrix,
    float scalar,
    size_t batch_size,
    size_t num_features,
    size_t threads_per_mini_batch,
    FeatureType* result) {

    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;

     // if(tidx==0&&bidx==0){
           
     //        printf("enter mateixmatrix multiplication\n");
  
     //    } 



    // size_t thread_offset = threadIdx.x % threads_per_datapoint;
    size_t num_thread_each_label = threads_per_mini_batch / LABEL_CLASS;
    //index relative to each label(corresponding to 784 parameter) 
    //Eg: 320 thread for 10 label -> each label 32 thread
    size_t tidx_label =  tidx / num_thread_each_label;
    size_t relative_tidx_label =  tidx % num_thread_each_label;

    if(tidx == 314 && blockIdx.x ==0){
        printf("%d -- %d\n",tidx_label,relative_tidx_label);
    }
    // asm("trap;");
    // strided sum of element-wise products concurrently in 10 dimentions
    // for (size_t j = relative_tidx_label; j < num_features; j+= num_thread_each_label)
    //     partial_dot += data_point_i[j] * parameter_vector[j + tidx_label * num_features];

    // result of the partial dot product is stored in shared memory
    // shared_memory[threadIdx.x] = partial_dot;



    // for(int m = 0 ; m < LABEL_CLASS ; m++){
        for (int j = 0; j < batch_size; j++) {
            for (int i = relative_tidx_label; i < num_features; i += threads_per_mini_batch) {
                // index of the point with respect to the whole dataset
                size_t point_idx = bidx * batch_size + j;
                // index of the feature with respect to all features in the dataset
                size_t feature_idx = point_idx * num_features + i;
                //gradient result 
                result[i+tidx_label*num_features] += datapoint_matrix[feature_idx] 
                    * probility_matrix[j+tidx_label*batch_size] * scalar;
            }
        }
    // }
}


// __device__ void d_matrixMatrixMultiply(
//     FeatureType* datapoint_matrix,
//     FeatureType* probility_matrix,
//     float scalar,
//     size_t batch_size,
//     size_t num_features,
//     size_t threads_per_mini_batch,
//     FeatureType* result) {

//     size_t tidx = threadIdx.x;
//     size_t bidx = blockIdx.x;

//      // if(tidx==0&&bidx==0){
           
//      //        printf("enter mateixmatrix multiplication\n");
  
//      //    } 
//     for(int m = 0 ; m < LABEL_CLASS ; m++){
//         for (int j = 0; j < batch_size; j++) {
//             for (int i = tidx; i < num_features; i += threads_per_mini_batch) {
//                 // index of the point with respect to the whole dataset
//                 size_t point_idx = bidx * batch_size + j;
//                 // index of the feature with respect to all features in the dataset
//                 size_t feature_idx = point_idx * num_features + i;
//                 //gradient result 
//                 result[i+m*num_features] += datapoint_matrix[feature_idx] 
//                     * probility_matrix[j+m*batch_size] * scalar;
//             }
//         }
//     }
// }





__device__ void d_matrixTranspose(
    FeatureType* probility_matrix,
    FeatureType* probility_transpose,
    size_t batch_size,
    size_t relative_tidx,
    size_t point_idx_in_block){
    
    //transpose from batch * Label to Label * batch
    if(relative_tidx < LABEL_CLASS){

        probility_transpose[relative_tidx*batch_size+point_idx_in_block] =
             probility_matrix[relative_tidx+point_idx_in_block*LABEL_CLASS];

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
    
    for (size_t i = tidx; i < num_features * LABEL_CLASS; i += threads_per_mini_batch) {
        FeatureType gradient_times_step_size = gradient[i] * step_size;
        atomicAdd(&parameter_vector[i], -gradient_times_step_size);
    }

}

// posibilily another way..
// speed almost the same
//  __device__ void d_updateParameters(
//     FeatureType* gradient,
//     FeatureType* parameter_vector,
//     size_t num_features,
//     size_t threads_per_mini_batch,
//     double step_size) {

//     // printf("enter update parameters in sgd_single_point\n");

//     // size_t thread_offset = threadIdx.x % threads_per_datapoint;


//     for(size_t i= 0;i<LABEL_CLASS;i++){
      
//         for (size_t j = threadIdx.x; j < num_features; j+=threads_per_mini_batch){
  
//             atomicAdd(&parameter_vector[j+i*num_features], -gradient[j+i*num_features]*step_size);
//         }        
//     }        
// }  

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



// one - way dimention parallel softmax function
__device__ void d_softMaxFunction1(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t point_idx_in_shmem,
    size_t relative_tidx,
    size_t point_idx_in_block,
    size_t num_label) {
    //copy (theta)T x and take fast exponential
    if(relative_tidx < num_label){
        posibility_each[point_idx_in_block * num_label+relative_tidx]
            = __expf(shared_memory[relative_tidx * blockDim.x+ point_idx_in_shmem]);
    }
    __syncthreads();

    //calculate sum , each thread has a copy (++)
    float sum = 0;
    for (size_t i=0;i<num_label;i++){
        sum += posibility_each[point_idx_in_block * num_label + i];
    }
    __syncthreads();
    
    //calculate final posibility for each point
    if(relative_tidx < num_label){
        posibility_each[point_idx_in_block * num_label+relative_tidx]/=sum;
    }
    __syncthreads();
}

//two - way dimention parallel softmax function
__device__ void d_softMaxFunction2(FeatureType* shared_memory, 
    FeatureType* posibility_each,
    size_t point_idx_in_shmem,
    size_t relative_tidx,
    size_t point_idx_in_block,
    size_t num_thread_each_label) {
    //copy (theta)T x and take fast exponential
    if(relative_tidx < LABEL_CLASS){
        posibility_each[point_idx_in_block * LABEL_CLASS+relative_tidx]
            = __expf(shared_memory[relative_tidx * num_thread_each_label + point_idx_in_shmem]);
    }
    __syncthreads();

    //calculate sum , each thread has a copy (++)
    float sum = 0;
    for (size_t i=0;i<LABEL_CLASS;i++){
        sum += posibility_each[point_idx_in_block * LABEL_CLASS + i];
    }
    __syncthreads();
    
    //calculate final posibility for each point
    if(relative_tidx < LABEL_CLASS){
        posibility_each[point_idx_in_block * LABEL_CLASS+relative_tidx]/=sum;
    }
    __syncthreads();
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
