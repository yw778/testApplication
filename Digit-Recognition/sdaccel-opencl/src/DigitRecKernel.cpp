//
//  DigitRecKernel.cpp
//  FFT
//
//  Created by Tim on 11/6/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//

#include "DigitRecKernel.h"
#include "training_data.h"
#include "testing_data.h"

DigitRecKernel::DigitRecKernel(const char *filename, const char* kernel_name, cl_device_type device_type) :
CLKernel(filename, kernel_name, device_type) {
    // empty child constructor
}

int DigitRecKernel::setup_kernel() {
    set_global(GLOBAL_WORK_GROUP_SIZE); //sets global work group size
    set_local(LOCAL_WORK_GROUP_SIZE);  //sets local work group size
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    return EXIT_SUCCESS;
}

int DigitRecKernel::load_data() {
    // Fill our data set with random float values
    int i = 0;
    for(i = 0; i < TRAINING_SIZE; i++) {
        training_set[i] = training_data[i];
    }
    
    for(i = 0; i < TEST_SIZE ; i++) {
        test_set[i] = testing_data[i];
    }
    
    // Create and set up the memory objects using the new CLMemObj class
    memobj_array.push_back(CLMemObj(training_set , sizeof(long long) , TRAINING_SIZE, CL_MEM_READ_ONLY));
    memobj_array.push_back(CLMemObj(test_set     , sizeof(long long) , TEST_SIZE    , CL_MEM_READ_ONLY));
    memobj_array.push_back(CLMemObj(results      , sizeof(long long) , RESULTS_SIZE , CL_MEM_WRITE_ONLY));
    
    return EXIT_SUCCESS;
}

bool DigitRecKernel::check_results() {
    // Read back the results from the device to verify the output
    read_clmem(2, sizeof(int), RESULTS_SIZE, results);
    
    // Validate our results
    unsigned int correct = 0; // number of correct results returned

    for(int i = 0; i < RESULTS_SIZE; i++) {
      if (results[i] == expected[i]) {
        correct++;
      } else {
        printf("results[%d] = %d, expected[%d] = %d\n", i, results[i], i, expected[i]);
      }

    }
    
    // Print a brief summary detailing the results
    printf("Computed '%d/%d' correct values!\n", correct, RESULTS_SIZE);
    
    return correct == RESULTS_SIZE ? true : false;
}

void DigitRecKernel::clean_up() {
    // Nothing to clean up for now
}
