#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "sgd_baseline.h"
#include "utils/spamfilter_utils.hpp"


// computes gradient for a single datapoint
static void gradientForSinglePoint(
    FeatureType* parameter_vector,
    FeatureType* data_point,
    LabelType label,
    size_t num_features,
    FeatureType* gradient) {

    //gradient for logistic function: x * (pi(theta, x) - y)
    double probability_of_positive =
        logisticFunction(parameter_vector, data_point, num_features);

    memset(gradient, 0, num_features * sizeof(FeatureType));

    addVectors(gradient,
               data_point,
               num_features,
               (probability_of_positive - label));
}

// executes serial implementation of stochastic gradient descent for
// logistic regression with a fixed number of iterations
// config_params: {step_size, characteristic_time}
void trainStochasticGradientDescent(
    DataSet training_set,
    TrainingOptions training_options) {

    // shuffle datapoints in order to add more stochasticity
    // shuffleKeyValue(training_set.data_points, training_set.labels,
    //                 training_set.num_data_points, training_set.num_features);

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

    double annealed_step_size = step_size;

    for (size_t k = 0; k < training_options.num_epochs; k++) {

        // simulated annealing (reduces step size as it converges)
        annealed_step_size = training_options.config_params["initial_step_size"]
                             / (1.0
                                + (curr_num_epochs
                                   * training_set.num_data_points
                                   / characteristic_time));
        curr_num_epochs++;

        for (size_t i = 0; i < training_set.num_data_points; i++) {
            // computes gradient
            gradientForSinglePoint(
                training_set.parameter_vector,
                &training_set.data_points[i * training_set.num_features],
                training_set.labels[i],
                training_set.num_features,
                gradient);

            // updates parameter vector
            updateParameters(
                training_set.parameter_vector,
                gradient,
                training_set.num_features,
                annealed_step_size);
        }
    }

    *training_options.step_size = annealed_step_size;

    delete[] gradient;
}
