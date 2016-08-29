//
//  main.cpp
//  OpenCLHarness
//
//  Created by Tim on 11/20/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "DigitRecKernel.h"

static void print_usage(void)
{
    std::cout << "usage: %s <options>\n";
    std::cout << "  -d <cpu|gpu|acc>\n";
    std::cout << "  -k [kernel name]\n";
    std::cout << "  -f [kernel fil]\n";
}

static int parse_command_line_args(int argc, char ** argv, cl_device_type * deviceType, std::string * kernelName , std::string *kernelFile) {
  int c = 0;

  while ((c = getopt(argc, argv, "d:k:f:")) != -1) {
    switch (c) {
      case 'd':
          if (strcmp(optarg, "gpu") == 0)
              *deviceType = CL_DEVICE_TYPE_GPU;
          else if (strcmp(optarg, "cpu") == 0)
              *deviceType = CL_DEVICE_TYPE_CPU;
          else if (strcmp(optarg, "acc") == 0) {
              *deviceType = CL_DEVICE_TYPE_ACCELERATOR;
          } else {
            print_usage();
            return -1;
          }
          break;
      case 'k':
          *kernelName = optarg;
          break;
      case 'f':
          *kernelFile = optarg;
          break;
      default:
          print_usage();
          return 1;
    } // matching on arguments
  } // while args present
}

int main(int argc, char ** argv) {
    cl_device_type deviceType = CL_DEVICE_TYPE_ACCELERATOR;
    std::string kernelName("");
    std::string kernelFile("");

    parse_command_line_args(argc, argv, &deviceType, &kernelName, &kernelFile);

    /******************OPEN CL STUFF************************/
    DigitRecKernel digitreckernel (kernelFile.c_str(), kernelName.c_str(), deviceType);
    fflush(stdout);
    digitreckernel.load_to_memory();
    fflush(stdout);
    digitreckernel.run_kernel();
    fflush(stdout);
    digitreckernel.check_results();
    digitreckernel.finish_and_clean_up();
    /******************END OF OPEN CL STUFF*****************/
    return 0;
}
