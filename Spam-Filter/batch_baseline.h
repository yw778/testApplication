#ifndef SPAMFILTER_BATCH_BASELINE
#define SPAMFILTER_BATCH_BASELINE

#include "spamfilter_defs.h"
#include <stdlib.h>

void trainBatchGradientDescent(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs,
    double tolerance = TOLERANCE,
    double step_size = STEP_SIZE,
    size_t num_points = NUM_TRAINING,
    size_t num_feats = NUM_FEATURES);

#endif
