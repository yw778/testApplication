#include <stdlib.h>

#include "spamfilter_fileio.hpp"
#include "spamfilter_defs.h"
#include "spamfilter_utils.hpp"

#include "baseline.h"
#include "batch_baseline.h"
#include "alt_one.h"

int main(int argc, char const *argv[]) {

    std::cout << "\nStart of Spam Filter Benchmark with " << NUM_RUNS << " runs averaged\n\n";

    // Model variables
    FeatureType* X = new FeatureType[X_SIZE]; // Data points
    LabelType* Y = new LabelType[Y_SIZE]; // Labels
    FeatureType* theta = new FeatureType[NUM_FEATURES]; // Model parameters

    // Read data from files and insert into variables
    const char points_filepath[] = "data/newfeats.dat";
    const char labels_filepath[] = "data/newlabels.dat";

    size_t readlines;
    if((readlines = readData(X_SIZE, points_filepath, X)) == 0) return 1;
    if((readlines = readData(Y_SIZE, labels_filepath, Y)) == 0) return 1;


    // Run each version of the training
    trainAndTest(trainStochasticGradientDescent, "SGD", X, Y, theta, NUM_EPOCHS, NUM_RUNS, STEP_SIZE);
    trainAndTest(trainBatchGradientDescent, "BGD",  X, Y, theta, NUM_EPOCHS, NUM_RUNS, STEP_SIZE);
    trainAndTest(trainParallelStochasticGradientDescent1, "Alt. #1",  X, Y, theta, NUM_EPOCHS, NUM_RUNS, STEP_SIZE);

    // Free memory and exit
    delete[] X;
    delete[] Y;
    delete[] theta;

    return 0;
}
