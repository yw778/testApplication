// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
#define NUM_FEATURES      1024/16
#define NUM_SAMPLES       100
#define NUM_TRAINING      90
#define NUM_TESTING       10
#define STEP_SIZE         50 //step size (eta)
#define NUM_EPOCHS        1
#define MAX_NUM_EPOCHS    1


typedef float FeatureType;
typedef float LabelType;
typedef float16 VectorFeatureType;
// #include "defs.h"
#define LOOP_PIPELINE __attribute__((xcl_pipeline_loop))
#define LOOP_UNROLL __attribute__((opencl_unroll_hint(16)))

/*
 * Parallel approach to Stochastic Gradient Descent #4 - Sdaccel - Opencl:
 *
 */


// dot product between two vectors
FeatureType cl_dotProduct(__local FeatureType* parameter_vector, __local FeatureType* data_point, int size) {

    VectorFeatureType result_vector = (0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f);

    // __local VectorFeatureType parameter_vector_16[NUM_FEATURES];

     // LOOP_UNROLL 
    // LOOP_PIPELINE
    // LOOP_UNROLL
    // for ( int j = 0; j < NUM_FEATURES ; j++ ) {

    //     // Read a new instance from the training set
    //     // VectorFeatureType parameter_instance;
    //     // parameter_vector_16[j].s0 = parameter_vector[j * 16    ];
    //     // parameter_vector_16[j].s1 = parameter_vector[j * 16 + 1];
    //     // parameter_vector_16[j].s2 = parameter_vector[j * 16 + 2];
    //     // parameter_vector_16[j].s3 = parameter_vector[j * 16 + 3];
    //     // parameter_vector_16[j].s4 = parameter_vector[j * 16 + 4];
    //     // parameter_vector_16[j].s5 = parameter_vector[j * 16 + 5];
    //     // parameter_vector_16[j].s6 = parameter_vector[j * 16 + 6];
    //     // parameter_vector_16[j].s7 = parameter_vector[j * 16 + 7];
    //     // parameter_vector_16[j].s8 = parameter_vector[j * 16 + 8];
    //     // parameter_vector_16[j].s9 = parameter_vector[j * 16 + 9];
    //     // parameter_vector_16[j].sa = parameter_vector[j * 16 + 10];
    //     // parameter_vector_16[j].sb = parameter_vector[j * 16 + 11];
    //     // parameter_vector_16[j].sc = parameter_vector[j * 16 + 12];
    //     // parameter_vector_16[j].sd = parameter_vector[j * 16 + 13];
    //     // parameter_vector_16[j].se = parameter_vector[j * 16 + 14];
    //     // parameter_vector_16[j].sf = parameter_vector[j * 16 + 15];
    //   }

    FeatureType result = 0;

    // LOOP_UNROLL  
    LOOP_PIPELINE
 LOOPA:   for (int j = 0; j < size; j++){       
            result_vector.s0 += parameter_vector[j*16] * data_point[j*16];
            result_vector.s1 += parameter_vector[j*16 + 1] * data_point[j*16 + 1];
            result_vector.s2 += parameter_vector[j*16 + 2] * data_point[j*16 + 2];
            result_vector.s3 += parameter_vector[j*16 + 3] * data_point[j*16 + 3];
            result_vector.s4 += parameter_vector[j*16 + 4] * data_point[j*16 + 4];
            result_vector.s5 += parameter_vector[j*16 + 5] * data_point[j*16 + 5];
            result_vector.s6 += parameter_vector[j*16 + 6] * data_point[j*16 + 6];
            result_vector.s7 += parameter_vector[j*16 + 7] * data_point[j*16 + 7];
            result_vector.s8 += parameter_vector[j*16 + 8] * data_point[j*16 + 8];
            result_vector.s9 += parameter_vector[j*16 + 9] * data_point[j*16 + 9];
            result_vector.sa += parameter_vector[j*16 + 10] * data_point[j*16 + 10];
            result_vector.sb += parameter_vector[j*16 + 11] * data_point[j*16 + 11];
            result_vector.sc += parameter_vector[j*16 + 12] * data_point[j*16 + 12];
            result_vector.sd += parameter_vector[j*16 + 13] * data_point[j*16 + 13];
            result_vector.se += parameter_vector[j*16 + 14] * data_point[j*16 + 14];
            result_vector.sf += parameter_vector[j*16 + 15] * data_point[j*16 + 15];
       
    }    
       
    // result_vector += parameter_vector_16[j] * data_point[j];

    result = result_vector.s0 + result_vector.s1 + result_vector.s2 + result_vector.s3
                + result_vector.s4 + result_vector.s5 + result_vector.s6 + result_vector.s7
                + result_vector.s8 + result_vector.s9 + result_vector.sa + result_vector.sb
                + result_vector.sc + result_vector.sd + result_vector.se + result_vector.sf;

    return result;

}

// hard logistic function
int cl_hardLogisticFunction(FeatureType exponent) {
    return (exponent >= 0) ? 1 : 0;
}

// logistic function
float cl_logisticFunction(FeatureType exponent) {
    return 1.0 / (1.0 + exp(-exponent));
}

// approximation of logistic function to avoid exponentiation
// float c_fauxLogisticFunction(FeatureType exponent) {

//     float e;

//     if (exponent < 0)
//         e = exponent * exponent + 1;
//     else
//         e = 1 / (exponent * exponent + 1);

//     return 1.0 / (1.0 + e);
// }

__attribute__ ((reqd_work_group_size(1, 1, 1)))
//__kernel void DigitRec(__global long long * global_training_set, __global long long * global_test_set, __global long long * global_results) {
__kernel void SgdLR(__global VectorFeatureType * global_data_points, 
    __global LabelType * global_labels, 
    __global VectorFeatureType * global_parameter_vector) {

    // event_t parameter_copy;
    // event_t results_copy;
    // event_t data_copy;
    event_t datacopy_evt[1];
    //TODO
    // Read data point from global memory
    __local FeatureType parameter_vector[NUM_FEATURES * 16];  __attribute__((xcl_array_partition(cyclic,16,1)));
    __local FeatureType data_point[NUM_FEATURES * NUM_TRAINING *16] __attribute__((xcl_array_partition(cyclic,16,1))); 
    // __attribute__((xcl_array_partition(cyclic,NUM_FEATURES,1)));
    __local FeatureType labels[NUM_TRAINING]; 
    // __attribute__((xcl_array_partition(complete, 1)));

    //TODO
    // datacopy_evt[0] = async_work_group_copy(parameter_vector, global_parameter_vector, NUM_FEATURES , 0);
    // LOOP_PIPELINE
    LOOP_UNROLL
   LOOPB: for (int i = 0; i < NUM_FEATURES ; i ++ )
    {
      VectorFeatureType tmp = global_parameter_vector[i];
      parameter_vector[i * 16 ] = tmp.s0;
      parameter_vector[i * 16 + 1] = tmp.s1;
      parameter_vector[i * 16 + 2] = tmp.s2;
      parameter_vector[i * 16 + 3] = tmp.s3;
      parameter_vector[i * 16 + 4] = tmp.s4;
      parameter_vector[i * 16 + 5] = tmp.s5;
      parameter_vector[i * 16 + 6] = tmp.s6;
      parameter_vector[i * 16 + 7] = tmp.s7;
      parameter_vector[i * 16 + 8] = tmp.s8;
      parameter_vector[i * 16 + 9] = tmp.s9;
      parameter_vector[i * 16 + 10] = tmp.sa;
      parameter_vector[i * 16 + 11] = tmp.sb;
      parameter_vector[i * 16 + 12] = tmp.sc;
      parameter_vector[i * 16 + 13] = tmp.sd;
      parameter_vector[i * 16 + 14] = tmp.se;
      parameter_vector[i * 16 + 15] = tmp.sf;
    }

    LOOP_UNROLL
  LOOPC:  for (int i = 0; i < NUM_FEATURES * NUM_TRAINING ; i ++ )
    {
      VectorFeatureType tmp1 = global_data_points[i];
      data_point[i * 16 ] = tmp1.s0;
      data_point[i * 16 + 1] = tmp1.s1;
      data_point[i * 16 + 2] = tmp1.s2;
      data_point[i * 16 + 3] = tmp1.s3;
      data_point[i * 16 + 4] = tmp1.s4;
      data_point[i * 16 + 5] = tmp1.s5;
      data_point[i * 16 + 6] = tmp1.s6;
      data_point[i * 16 + 7] = tmp1.s7;
      data_point[i * 16 + 8] = tmp1.s8;
      data_point[i * 16 + 9] = tmp1.s9;
      data_point[i * 16 + 10] = tmp1.sa;
      data_point[i * 16 + 11] = tmp1.sb;
      data_point[i * 16 + 12] = tmp1.sc;
      data_point[i * 16 + 13] = tmp1.sd;
      data_point[i * 16 + 14] = tmp1.se;
      data_point[i * 16 + 15] = tmp1.sf;
    }

    // datacopy_evt[0] = async_work_group_copy(data_point, global_data_points, NUM_FEATURES * NUM_TRAINING , 0);
    datacopy_evt[0] = async_work_group_copy(labels, global_labels, NUM_TRAINING, 0);
    // wait_group_events(1, &data_copy);
    // wait_group_events(1, &parameter_copy);
    wait_group_events(1, datacopy_evt);

    // barrier(CLK_LOCAL_MEM_FENCE);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        // Iterate over all training instances (data points)
        // static int read = 0;
        // LOOP_PIPELINE
        for (int i = 0; i < NUM_TRAINING; i++) {

            // starts computation of gradient
            FeatureType dot = cl_dotProduct(parameter_vector, &data_point[i * NUM_FEATURES * 16], NUM_FEATURES);

            float probability_of_positive = cl_hardLogisticFunction(dot);   
            //TODO
            float step = -(probability_of_positive - labels[i]) * STEP_SIZE;

            // finishes computation of (gradient * step size) and updates parameter vector
            // LOOP_PIPELINE
            // LOOP_UNROLL
            LOOP_PIPELINE 
        LOOPD:    for (int j = 0; j < NUM_FEATURES; j++){
                parameter_vector[j * 16] += step * data_point[(i * NUM_FEATURES + j) * 16];
                parameter_vector[j * 16 + 1] += step * data_point[(i * NUM_FEATURES + j) * 16 + 1];
                parameter_vector[j * 16 + 2] += step * data_point[(i * NUM_FEATURES + j) * 16 + 2];
                parameter_vector[j * 16 + 3] += step * data_point[(i * NUM_FEATURES + j) * 16 + 3];
                parameter_vector[j * 16 + 4] += step * data_point[(i * NUM_FEATURES + j) * 16 + 4];
                parameter_vector[j * 16 + 5] += step * data_point[(i * NUM_FEATURES + j) * 16 + 5];
                parameter_vector[j * 16 + 6] += step * data_point[(i * NUM_FEATURES + j) * 16 + 6];
                parameter_vector[j * 16 + 7] += step * data_point[(i * NUM_FEATURES + j) * 16 + 7];
                parameter_vector[j * 16 + 8] += step * data_point[(i * NUM_FEATURES + j) * 16 + 8];
                parameter_vector[j * 16 + 9] += step * data_point[(i * NUM_FEATURES + j) * 16 + 9];
                parameter_vector[j * 16 + 10] += step * data_point[(i * NUM_FEATURES + j) * 16 + 10];
                parameter_vector[j * 16 + 11] += step * data_point[(i * NUM_FEATURES + j) * 16 + 11];
                parameter_vector[j * 16 + 12] += step * data_point[(i * NUM_FEATURES + j) * 16 + 12];
                parameter_vector[j * 16 + 13] += step * data_point[(i * NUM_FEATURES + j) * 16 + 13];
                parameter_vector[j * 16 + 14] += step * data_point[(i * NUM_FEATURES + j) * 16 + 14];
                parameter_vector[j * 16 + 15] += step * data_point[(i * NUM_FEATURES + j) * 16 + 15];
                // parameter_vector[j] += step * data_point[i * NUM_FEATURES * 16 + j];
            }
        }
    }

    // LOOP_PIPELINE
    LOOP_UNROLL
LOOPE:    for ( int j = 0; j < NUM_FEATURES; j++ ) {

        // Read a new instance from the training set
        // VectorFeatureType parameter_instance;
        global_parameter_vector[j].s0 = parameter_vector[j * 16    ];
        global_parameter_vector[j].s1 = parameter_vector[j * 16 + 1];
        global_parameter_vector[j].s2 = parameter_vector[j * 16 + 2];
        global_parameter_vector[j].s3 = parameter_vector[j * 16 + 3];
        global_parameter_vector[j].s4 = parameter_vector[j * 16 + 4];
        global_parameter_vector[j].s5 = parameter_vector[j * 16 + 5];
        global_parameter_vector[j].s6 = parameter_vector[j * 16 + 6];
        global_parameter_vector[j].s7 = parameter_vector[j * 16 + 7];
        global_parameter_vector[j].s8 = parameter_vector[j * 16 + 8];
        global_parameter_vector[j].s9 = parameter_vector[j * 16 + 9];
        global_parameter_vector[j].sa = parameter_vector[j * 16 + 10];
        global_parameter_vector[j].sb = parameter_vector[j * 16 + 11];
        global_parameter_vector[j].sc = parameter_vector[j * 16 + 12];
        global_parameter_vector[j].sd = parameter_vector[j * 16 + 13];
        global_parameter_vector[j].se = parameter_vector[j * 16 + 14];
        global_parameter_vector[j].sf = parameter_vector[j * 16 + 15];
    }

    // barrier(CLK_LOCAL_MEM_FENCE);
    // event_t result_evt = async_work_group_copy(global_parameter_vector, parameter_vector, NUM_FEATURES * 16, 0);
    // wait_group_events(1, &result_evt);
    // barrier(CLK_LOCAL_MEM_FENCE);
    // wait_group_events(1, &results_copy);
}

