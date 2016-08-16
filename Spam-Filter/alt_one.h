#ifndef SPAMFILTER_ALT_ONE
#define SPAMFILTER_ALT_ONE

/*
 * Parallel approach #1:
 * Using OpenBLAS for regular stochastic gradient descent
 */

 #include "spamfilter_defs.h"
 #include <stdlib.h>

 void trainParallelStochasticGradientDescent1(
     FeatureType* X,
     LabelType* Y,
     FeatureType* theta,
     size_t max_num_epochs,
     double tolerance = TOLERANCE,
     double step_size = STEP_SIZE,
     size_t num_points = NUM_TRAINING,
     size_t num_feats = NUM_FEATURES);

#endif
