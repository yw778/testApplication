#include <math.h>


#include "spamfilter_utils.hpp"
#include "spamfilter_timer.h"

#include "spamfilter_fileio.hpp"

// adds two vectors and stores the results in the first one
void add_vectors(float* a, float* b, const size_t size, const float scale_for_a) {
    cblas_saxpy(size, scale_for_a, b, 1, a, 1);
}

// computes dot product for two given vectors a and b
float dot_product(float* a, float* b, const size_t size) {
    return cblas_sdot(size, a, 1, b, 1);
}

// computes norm 2 of a given vector v
float norm2(float* v, const size_t size) {
    return cblas_snrm2(size, v, 1);
}

// computes logistic function for a given double
double sigmoid(double exponent) {
    return (1.0 / (1.0 + exp(-exponent)));
}

// computes logistic function for a given parameter vector (theta) and a data point (x_i)
double sigmoid(FeatureType* theta, FeatureType* x_i, const size_t num_feats) {
    return sigmoid(dot_product(theta, x_i, num_feats));
}

// matrix-vector multiplication wrapper
void matrixVectorMultiply(float* A, float* v, float alpha, size_t num_points, size_t num_feats, float* result) {
    cblas_sgemv(CblasRowMajor, CblasTrans, num_points, num_feats, alpha, A, num_feats, v, 1, 0, result, 1);
}


// updates the parameters (theta)
void updateParameters(FeatureType* theta, FeatureType* gradient, size_t num_feats, double step_size, bool revert) {
    double sign = revert ? 1 : -1;
    step_size *= sign;
    cblas_saxpy (num_feats, step_size, gradient, 1, theta, 1);
}

// gets predicted label for a given parameter vector (theta) and a given datapoint (x_i)
LabelType getPrediction(FeatureType* theta, FeatureType* x_i, size_t num_feats) {
    float theta_dot_x_i = dot_product(theta, x_i, num_feats);
    return (theta_dot_x_i > 0) ? 1 : 0;
}

// serial implementation of accuracy test
double computeErrorRate(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t beginning,
    size_t end,
    size_t num_feats) {

    size_t error_count = 0;
    for (size_t i = beginning; i < end; i++) {
        LabelType prediction = getPrediction(theta, &X[i*num_feats], num_feats);
        if (prediction != Y[i])
            error_count++;
    }
    size_t num_tested_samples = end - beginning;
    return (double) error_count / num_tested_samples;
}

// loss function for the whole batch (Negative LogLikelihood for Bernoulli probability distribution)
double lossFunction(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t num_points,
    size_t num_feats) {

    double sum = 0.0;
    for (size_t i = 0; i < num_points; i++) {
        double probability_of_positive = sigmoid(theta, &X[i * num_feats], num_feats);
        FeatureType y_i = (FeatureType) Y[i];
        sum += ((y_i * log(probability_of_positive)) + ((1 - y_i) * log(1 - probability_of_positive))) / num_points;
    }
    return -sum; // negative of the sum
}

void trainAndTest(
    void (*func)(FeatureType*, LabelType*, FeatureType*, size_t, double, double, size_t, size_t),
    const char* name,
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs,
    unsigned int num_runs,
    double step_size,
    double tolerance) {

    // Measure training time
    Timer training_timer(name);
    Timer full_cycle_timer("Full cycle");

    // Initialize benchmark variables
    double average_training_time = 0.0;
    double average_full_cycle_time = 0.0;
    //double average_loss = 0.0;
    double average_training_error = 0.0;
    double average_testing_error = 0.0;

    // Shuffle vectors and train
    for (size_t k = 0; k < num_runs; k++) {
        full_cycle_timer.start();
        memset(theta, 0, NUM_FEATURES * sizeof(FeatureType));

        shuffleKeyValue(X, Y, NUM_SAMPLES, NUM_FEATURES);
        training_timer.start();
        func(X, Y, theta, max_num_epochs, tolerance, step_size, NUM_TRAINING, NUM_FEATURES);
        average_training_time += training_timer.stop();
        average_full_cycle_time += full_cycle_timer.stop();

        // Get loss function for the whole training batch
        //average_loss += lossFunction(X, Y, theta, NUM_TRAINING, NUM_FEATURES);

        // Get Training error
        average_training_error += computeErrorRate(X, Y, theta, 0, NUM_TRAINING);

        // Get Testing error
        average_testing_error += computeErrorRate(X, Y, theta, NUM_TRAINING, (NUM_TRAINING + NUM_TESTING));
    }

    average_training_time /= num_runs;
    average_full_cycle_time /= num_runs;
    //average_loss /= num_runs;
    average_training_error /= num_runs;
    average_testing_error /= num_runs;

    // Output error rates and loss
    printf("Average time for training of " ANSI_COLOR_GREEN "\"%s\"" ANSI_COLOR_RESET " (msecs):" ANSI_COLOR_YELLOW " %f\n" ANSI_COLOR_RESET, name, average_training_time);
    printf("Average time for shuffling and training of " ANSI_COLOR_GREEN "\"%s\"" ANSI_COLOR_RESET " (msecs):" ANSI_COLOR_YELLOW " %f\n" ANSI_COLOR_RESET, name, average_training_time);
    //printf("Average loss for training batch: %f\n", average_loss);
    printf("Average training error: " ANSI_COLOR_RED "%.3f %%" ANSI_COLOR_RESET "\n", average_training_error * 100);
    printf("Average testing error: " ANSI_COLOR_RED "%.3f %%" ANSI_COLOR_RESET "\n", average_testing_error * 100);
}
