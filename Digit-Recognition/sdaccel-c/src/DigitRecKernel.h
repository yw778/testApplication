//
//  VVaddKernel.h
//  FFT
//
//  Created by Tim on 11/6/15.
//  Copyright (c) 2015 Tim. All rights reserved.

#ifndef __FFT__VVaddKernel__
#define __FFT__VVaddKernel__

#include <stdio.h>
#include "../../../harness/sdaccel/CLKernel.h"

class DigitRecKernel : public CLKernel {
    static const unsigned int GLOBAL_WORK_GROUP_SIZE = 1;
    static const unsigned int LOCAL_WORK_GROUP_SIZE  = 1;
    static const unsigned int TEST_SIZE     = 2000  * 4;
    static const unsigned int TRAINING_SIZE = 18000 * 4;
    static const unsigned int RESULTS_SIZE  = 2000;
    long long test_set     [TEST_SIZE];     // original data set given to device
    long long training_set [TRAINING_SIZE]; // original data set given to device
    int results      [RESULTS_SIZE];  // results returned from device
public:
    DigitRecKernel(const char *filename, const char* kernel_name, cl_device_type device_type);
    int setup_kernel ();
    int load_data();
    bool check_results();
    void clean_up();
};

#endif /* defined(__FFT__VVaddKernel__) */
