#ifndef SPAMFILTER_UTILS
#define SPAMFILTER_UTILS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cblas.h>

#include "spamfilter_defs.h"

double sigmoid(double exponent);
double sigmoid(FeatureType* theta, FeatureType* x_i, const size_t num_feats);

void add_vectors(float* a, float* b, const size_t size, const float scale_for_a = 1);

float dot_product(float* a, float* b, const size_t size);

void matrixVectorMultiply(float* A, float* v, float alpha, size_t num_points, size_t num_feats, float* result);

float norm2(float* v, const size_t size);

// swap elements in an array
template <typename Type>
void swap(Type *array, size_t index1, size_t index2, const size_t elements_per_chunk = 1) {
    Type* temp = new Type[elements_per_chunk];
    index1 *= elements_per_chunk;
    index2 *= elements_per_chunk;
    const size_t size_of_chunk = elements_per_chunk * sizeof(Type);
    memcpy(temp, &array[index1], size_of_chunk);
    memcpy(&array[index1], &array[index2], size_of_chunk);
    memcpy(&array[index2], temp, size_of_chunk);
    delete[] temp;
}

// shuffle array randomly
template <typename KeyType, typename ValueType>
void shuffleKeyValue(KeyType *key_array, ValueType *value_array, size_t size, const size_t key_elements_per_chunk = 1, const size_t value_elements_per_chunk = 1) {
    //can't use % 0
    if (size == 0) {
        return;
    }

    srand(static_cast<unsigned int>(time(0)));
    size_t j;
    for (size_t i = 0; i < size; i++) {
        j = rand() % size;
        swap(key_array, i, j, key_elements_per_chunk);
        swap(value_array, i, j, value_elements_per_chunk);
    }
}

void updateParameters(
    FeatureType* theta,
    FeatureType* gradient,
    size_t num_feats,
    double step_size,
    bool revert = false);

double computeErrorRate(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t beginning = 0,
    size_t end = NUM_TRAINING,
    size_t num_feats = NUM_FEATURES);

double lossFunction(
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t num_points,
    size_t num_feats);

void trainAndTest(
    void (*func)(FeatureType*, LabelType*, FeatureType*, size_t, double, double, size_t, size_t),
    const char* name,
    FeatureType* X,
    LabelType* Y,
    FeatureType* theta,
    size_t max_num_epochs = NUM_EPOCHS,
    unsigned int num_runs = NUM_RUNS,
    double step_size = STEP_SIZE,
    double tolerance = TOLERANCE);

#endif
