// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
#define NUM_FEATURES      1024
#define NUM_SAMPLES       100
#define NUM_TRAINING      90
#define NUM_TESTING       10
#define STEP_SIZE         50 //step size (eta)
#define NUM_EPOCHS        1
#define MAX_NUM_EPOCHS    1


typedef float FeatureType;
typedef float LabelType;
// #include "defs.h"
#define LOOP_PIPELINE __attribute__((xcl_pipeline_loop))
#define LOOP_UNROLL __attribute__((opencl_unroll_hint))

// #define PRAGMA_SUB(x) _Pragma (#x)
// #define DO_PRAGMA(x) PRAGMA_SUB(x)

/*
 * Parallel approach to Stochastic Gradient Descent #3 - Vivado HLS - C:
 *
 */

// needs this when the kernel is in a cpp file
// extern "C" {

// dot product between two vectors
FeatureType cl_dotProduct(__local FeatureType* a, __local FeatureType* b, int size) {

    FeatureType result = 0;

    LOOP_PIPELINE
    for (int j = 0; j < size; j++)
        result += a[j] * b[j];

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

// Top-level Kernel
// Stochastic Gradient Descent for Logistic Regression
// void SgdLR(
//     FeatureType* global_data_points,
//     LabelType* global_labels,
//     FeatureType* global_parameter_vector) {

//     // DO_PRAGMA(HLS INTERFACE ap_bus port=global_data_points offset=slave bundle=gmem depth=DATA_SET_SIZE)
//     // DO_PRAGMA(HLS INTERFACE ap_bus port=global_labels offset=slave bundle=gmem depth=NUM_SAMPLES)
//     // DO_PRAGMA(HLS INTERFACE ap_bus port=global_parameter_vector offset=slave bundle=gmem depth=NUM_FEATURES)
//     DO_PRAGMA(HLS INTERFACE m_axi port=global_data_points offset=slave bundle=gmem0 depth=DATA_SET_SIZE)
//     DO_PRAGMA(HLS INTERFACE m_axi port=global_labels offset=slave bundle=gmem1 depth=NUM_SAMPLES)
//     DO_PRAGMA(HLS INTERFACE m_axi port=global_parameter_vector offset=slave bundle=gmem2 depth=NUM_FEATURES)
//     DO_PRAGMA(HLS INTERFACE s_axilite port=global_data_points bundle=control)
//     DO_PRAGMA(HLS INTERFACE s_axilite port=global_labels bundle=control)
//     DO_PRAGMA(HLS INTERFACE s_axilite port=global_parameter_vector bundle=control)
//     DO_PRAGMA(HLS INTERFACE s_axilite port=return bundle=control)

//     printf("Entered kernel\n"); // for debugging

//     // Device vectors
//     FeatureType data_point_i[NUM_FEATURES];
//     FeatureType parameter_vector[NUM_FEATURES];

//     // Auxiliar variables for cycles' indexes
//     size_t i;
//     size_t j;
//     size_t epoch;

//     // Read parameter vector from global memory
//     for (j = 0; j < NUM_FEATURES; j++)
//         parameter_vector[j] = global_parameter_vector[j];

//     for (epoch = 0; epoch < NUM_EPOCHS; epoch++) {

//         // Iterate over all training instances (data points)
//         static int read = 0;
//         for (i = 0; i < NUM_TRAINING; i++) {
//             // Read data point from global memory
//             read = 0;
//             for (j = 0; j < NUM_FEATURES; j++)
//                 data_point_i[j] = global_data_points[j + i * NUM_FEATURES];

//             // starts computation of gradient
//             FeatureType dot = c_dotProduct(parameter_vector, data_point_i, NUM_FEATURES);

//             float probability_of_positive = c_hardLogisticFunction(dot);

//             float step = -(probability_of_positive - global_labels[i]) * STEP_SIZE;

//             // finishes computation of (gradient * step size) and updates parameter vector
//             for (j = 0; j < NUM_FEATURES; j++)
//                 parameter_vector[j] += step * data_point_i[j];

//         }
//     }

//     // copy results back to global memory
//     for (j = 0; j < NUM_FEATURES; j++)
//         global_parameter_vector[j] = parameter_vector[j];

//     printf("Exited kernel\n"); // for debugging

//     return;
// }


// } // end of extern C

__attribute__ ((reqd_work_group_size(1, 1, 1)))
//__kernel void DigitRec(__global long long * global_training_set, __global long long * global_test_set, __global long long * global_results) {
__kernel void SgdLR(__global FeatureType * global_data_points, 
    __global LabelType * global_labels, 
    __global FeatureType * global_parameter_vector) {

    event_t parameter_copy;
    event_t results_copy;
    // event_t data_copy;

    __local FeatureType parameter_vector[NUM_FEATURES]; 
    __local FeatureType data_point_i[NUM_FEATURES];

    parameter_copy = async_work_group_copy(parameter_vector, global_parameter_vector, NUM_FEATURES , 0);
    wait_group_events(1, &parameter_copy);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        // Iterate over all training instances (data points)
        // static int read = 0;
        LOOP_PIPELINE
        for (int i = 0; i < NUM_TRAINING; i++) {
            // Read data point from global memory
            // read = 0;
            event_t data_copy;
            data_copy = async_work_group_copy(data_point_i, &global_data_points[i * NUM_FEATURES], NUM_FEATURES , 0);
            wait_group_events(1, &data_copy);

            // for (int j = 0; j < NUM_FEATURES; j++)
            //     data_point_i[j] = global_data_points[j + i * NUM_FEATURES];

            // starts computation of gradient
            FeatureType dot = cl_dotProduct(parameter_vector, data_point_i, NUM_FEATURES);

            float probability_of_positive = cl_hardLogisticFunction(dot);

            float step = -(probability_of_positive - global_labels[i]) * STEP_SIZE;

            // finishes computation of (gradient * step size) and updates parameter vector
            LOOP_PIPELINE
            for (int j = 0; j < NUM_FEATURES; j++)
                parameter_vector[j] += step * data_point_i[j];

        }
    }

    results_copy = async_work_group_copy(global_parameter_vector, parameter_vector, NUM_FEATURES, 0);
    wait_group_events(1, &results_copy);
}

