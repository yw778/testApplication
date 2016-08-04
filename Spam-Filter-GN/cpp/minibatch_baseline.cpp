#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "minibatch_baseline.h"
#include "utils/spamfilter_utils.hpp"

/* This is a baseline for stochastic gradient descent using mini-batches. In this version,
the mini-batch parameters are updated only once in an epoch until they all converge. This version runs faster
than when each mini-batch goes through all the epochs before the next mini-batch is computed. */


template <typename Type1, typename Type2, typename Type3, typename Type4>
static void nBlas_matrixVectorMultiply(Type1* matrix, 
                                       Type2* vect, 
                                       Type3 scalar, 
                                       size_t num_data_points, 
                                       size_t num_features, 
                                       Type4* result) {
    // non-blas function for matrixVectorMultiplication
    memset(result, 0, num_features * sizeof(Type4));
    for (size_t i = 0; i < num_data_points; i++) {
        addVectors(result, &matrix[i * num_features], num_features, scalar * vect[i]);
    }
}


static void gradientForMiniBatch(
    DataSet training_set,
    FeatureType* gradient,
    size_t batch_size,
    size_t starting_point) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType));

    float* probabilities_of_positive = new float[batch_size];
    size_t idx = 0;

    //computes logistic function for each data point in the mini-batch
    for (int i= starting_point; i < starting_point + batch_size; i++) {
        idx = i * training_set.num_features;
        probabilities_of_positive[i % batch_size] 
            = logisticFunction(training_set.parameter_vector, 
                               &training_set.data_points[idx], 
                               training_set.num_features);
    }

    // computes difference between predicted probability and actual label (PI - Y)
    addVectors(probabilities_of_positive, 
               &training_set.labels[starting_point], 
               batch_size, -1);

    // finishes computation of gradient: (1/n) * X^T * (PI(theta, X) - YI)
    float factor = 1.0f/batch_size;
    size_t starting_point_idx = starting_point *  training_set.num_features;
    matrixVectorMultiply(&training_set.data_points[starting_point_idx], 
                         probabilities_of_positive, factor, batch_size, 
                         training_set.num_features, gradient);

    delete[] probabilities_of_positive;
}

void trainMiniBatchGradientDescent(
    DataSet training_set,
    TrainingOptions training_options) {

    FeatureType* gradient = new FeatureType[training_set.num_features];

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

    const size_t batch_size =
            (fieldExists(training_options.config_params, "batch_size"))
            ? training_options.config_params["batch_size"]
            : BATCH_SIZE; 
    
    size_t num_mini_batches = training_set.num_data_points / batch_size;

    double annealed_step_size = step_size;

    for (int i = 0; i < training_options.num_epochs; i++){
    // simulated annealing (reduces step size as it converges)
        annealed_step_size = training_options.config_params["initial_step_size"]
                             / (1.0
                                + (curr_num_epochs
                                   * training_set.num_data_points
                                   / characteristic_time));
        curr_num_epochs++;

        for (int j = 0; j < num_mini_batches; j++){
            size_t idx = j * batch_size;
            // compute gradient
            gradientForMiniBatch(training_set, gradient, batch_size, idx);

            // update parameters
            updateParameters(training_set.parameter_vector, gradient, 
                training_set.num_features, annealed_step_size);
        }
    }

    *training_options.step_size = annealed_step_size;

    delete[] gradient;

}
