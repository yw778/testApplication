#include <stdio.h>
#include <string.h>
#include <math.h>

#include "batch_baseline.h"
#include "spamfilter_utils.hpp"


// computes gradient for the whole training set
static void gradientForWholeBatch(
    FeatureType* theta,
    FeatureType* X,
    LabelType* Y,
    size_t num_points,
    size_t num_feats,
    FeatureType* gradient) {

    memset(gradient, 0, num_feats * sizeof(FeatureType));

    float* pi = new float[num_points];

    size_t idx = 0;
    for (size_t i = 0; i < num_points; i++) {
        idx = i * num_feats;
        pi[i] = sigmoid(theta, &X[idx], num_feats);
    }

    add_vectors(pi, Y, num_points, -1);
    float one_over_n = 1.0f / num_points;
    matrixVectorMultiply(X, pi, one_over_n, num_points, num_feats, gradient);

    delete[] pi;
}

// Bold Driver approach
static void boldDriver(
    double previous_loss,
    double current_loss,
    FeatureType* theta,
    FeatureType* gradient,
    size_t num_feats,
    double* step_size) {
    if (current_loss < previous_loss) {
        *step_size *= 1.045;
    } else {
        bool revert = true;
        updateParameters(theta, gradient, num_feats, *step_size, revert);
        *step_size *= 0.5;
    }
}

// executes serial implementation of stochastic gradient descent for logistic regression until convergence or for a fixed number of epochs
void trainBatchGradientDescent(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs,
    double tolerance,
    double step_size,
    size_t num_points,
    size_t num_feats) {

    FeatureType* gradient = new FeatureType[num_feats];

    double previous_loss = lossFunction(X, Y, theta, num_points, num_feats);
    double current_loss;

    for (size_t i = 0; i < max_num_epochs; i++) {
        gradientForWholeBatch(theta, X, Y, num_points, num_feats, gradient);
        updateParameters(theta, gradient, num_feats, step_size);

        current_loss = lossFunction(X, Y, theta, num_points, num_feats);
        boldDriver(previous_loss, current_loss, theta, gradient, num_feats, &step_size);
        previous_loss = current_loss;

        if (norm2(gradient, num_feats) < tolerance)
            break;
    }

    delete[] gradient;

}
