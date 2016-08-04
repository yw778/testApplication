#ifndef SPAMFILTER_SGD_CUBLAS
#define SPAMFILTER_SGD_CUBLAS

/*
 * Parallel approach #1:
 * Using OpenBLAS for regular stochastic gradient descent
 */

#include "utils/spamfilter_defs.h"

void trainStochasticGradientDescent1(DataSet training_set, TrainingOptions training_options);

#endif
