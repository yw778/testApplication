#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "batch_baseline.h"
#include "utils/spamfilter_utils.hpp"


// computes gradient for the whole training set
static void gradientForWholeBatch(
    DataSet training_set,
    FeatureType* gradient) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType));

    float* probabilities_of_positive = new float[training_set.num_data_points];

    // computes logistic function for each data point in the training set
    size_t idx = 0;
    for (size_t i = 0; i < training_set.num_data_points; i++) {

        idx = i * training_set.num_features;

        probabilities_of_positive[i] =  logisticFunction(
                                                training_set.parameter_vector,
                                                &training_set.data_points[idx],
                                                training_set.num_features);
    }

    // computes difference between
    // predicted probability and actual label: (PI - Y)
    addVectors(probabilities_of_positive,
               training_set.labels,
               training_set.num_data_points,
               -1);

    // finishes computation of gradient: (1/n) * X^T * (PI(theta, X) - YI)
    float factor = 1.0f / training_set.num_data_points;
    matrixVectorMultiply(training_set.data_points,
                         probabilities_of_positive,
                         factor,
                         training_set.num_data_points,
                         training_set.num_features,
                         gradient);

    delete[] probabilities_of_positive;
}

// Bold Driver: adjusting the step size according to the result of the
// last step and reverting the step if results are worse than they were before.
static void boldDriver(
    DataSet training_set,
    FeatureType* gradient,
    double* step_size) {

    double previous_loss = lossFunction(training_set);

    updateParameters(training_set.parameter_vector,
                     gradient,
                     training_set.num_features,
                     *step_size);

    double current_loss = lossFunction(training_set);

    // if it's going in the right direction, increase step size
    if (current_loss < previous_loss) {
        *step_size *= 1.045;
    }
    // if the previous step was too big and the loss increased,
    // revert step and reduce step size
    else {
        bool revert = true;
        int num_reverts = 0, max_reverts = 10;
        while (revert && (num_reverts < max_reverts)) {
            updateParameters(training_set.parameter_vector,
                             gradient,
                             training_set.num_features,
                             *step_size,
                             revert);

            *step_size *= 0.5;

            updateParameters(training_set.parameter_vector,
                             gradient,
                             training_set.num_features,
                             *step_size);

             current_loss = lossFunction(training_set);

             revert = (current_loss > previous_loss);
        }

    }
}

// uses simulated annealing to update step size
static void anneal(
    size_t current_epoch,
    double initial_step_size,
    double* annealed_step_size,
    DataSet training_set,
    FeatureType* gradient) {

    double characteristic_time = CHARACTERISTIC_TIME;

    (*annealed_step_size) = initial_step_size
                          / (1.0 + (current_epoch / characteristic_time));

    updateParameters(training_set.parameter_vector,
                   gradient,
                   training_set.num_features,
                   *annealed_step_size);
}

// executes serial implementation of stochastic gradient descent for
// logistic regression until convergence or for a fixed number of epochs
void trainBatchGradientDescent(
    DataSet training_set,
    TrainingOptions training_options) {

    // read configuration parameters
    double step_size = *training_options.step_size;
    double initial_step_size = step_size;

    const double tolerance =
            (fieldExists(training_options.config_params, "tolerance"))
            ? training_options.config_params["tolerance"]
            : TOLERANCE;

    FeatureType* gradient = new FeatureType[training_set.num_features];


    for (size_t epoch = 0; epoch < training_options.num_epochs; epoch++) {
        // compute gradient and update parameters
        gradientForWholeBatch(training_set, gradient);

        boldDriver(training_set, gradient, &step_size);
        // anneal(epoch, initial_step_size, &step_size, training_set, gradient);

        // stop iterating if the gradient is close to zero
        if (norm2(gradient, training_set.num_features) < tolerance)
            break;
    }

    *training_options.step_size = step_size;

    delete[] gradient;
}
