#include <stdio.h>
#include <string.h>

#include "baseline.h"
#include "spamfilter_utils.hpp"


// computes gradient for a single datapoint
static void gradientForSinglePoint(FeatureType* theta, FeatureType* x_i, LabelType y, size_t num_feats, FeatureType* gradient) {
    double probability_of_positive = sigmoid(theta, x_i, num_feats);
    double pi_minus_yi = probability_of_positive - y;
    memset(gradient, 0, num_feats * sizeof(FeatureType));
    add_vectors(gradient, x_i, num_feats, pi_minus_yi);
}

// executes serial implementation of stochastic gradient descent for logistic regression with a fixed number of iterations
void trainStochasticGradientDescent(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs,
    double tolerance,
    double step_size,
    size_t num_points,
    size_t num_feats){

    FeatureType* gradient = new FeatureType[num_feats];

    double annealed_step_size;
    const double characteristic_time = max_num_epochs * num_points / 3;
    size_t curr_num_iterations;

    for (size_t k = 0; k < max_num_epochs; k++) {
        for (size_t i = 0; i < num_points; i++) {
            gradientForSinglePoint(theta, &X[i * num_feats], Y[i], num_feats, gradient);
            curr_num_iterations = k * num_points + i;
            annealed_step_size = step_size / (1.0 + (curr_num_iterations / characteristic_time));
            updateParameters(theta, gradient, num_feats, annealed_step_size);
        }
    }

    delete[] gradient;
}
