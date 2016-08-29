# Digit Recognition Tutorial

This tutorial goes over how to write, build and run the OpenCL Digit Recognition Application.

## Setup Environment

Based on the environment source one of the setup scripts in the top level directory.

1. `source setup-zhang-01.sh`
2. `source setup-zhang-precision.sh`

## Designing an application

This section is a general overview of how to write an application.

### Main
[main](main.cpp) configures the kernel class. The following code generally configures the class.

```cpp
DigitRecKernel digitreckernel (kernelFile.c_str(), kernelName.c_str(), deviceType, 3);
digitreckernel.load_to_memory();
digitreckernel.run_kernel();
digitreckernel.check_results();
```

Apart from the name of the Kernel Class (```DigitRecKernel```) the user only has to change the number of memory units (3 for digit recognition) which correpsond to the number of arguments in the OpenCL Kernel.

### Header

[DigitRecKernel.h](DigitRecKernel.h) is the header for the Kernel class. Below are the two main ```static const``` variables that must be present for all Kernel classes.

```cpp
static const unsigned int GLOBAL_WORK_GROUP_SIZE = 1;
static const unsigned int LOCAL_WORK_GROUP_SIZE  = 1;
```

### Kernel Class Implementation

[DigitRecKernel.cpp](DigitRecKernel.cpp) implements the source for the Kernel class. The constructor should be empty. For now ```DigitRecKernel::setup_kernel()``` should only set the ```GLOBAL_WORK_GROUP_SIZE``` and ```LOCAL_WORK_GROUP_SIZE```. The following functions should be implemented accordingly

```cpp
int DigitRecKernel::load_data();
bool DigitRecKernel::check_results();
```

### Kernel Code

[digit_rec_kernel.cl](digit_rec_kernel.cl) implements the OpenCL kernel code.

## Software Emulation

The software emulation runs the OpenCL kernels on the host CPU. This is useful
for functional verification. The SW emulation is fast to build (seconds) and
run. This enables agile software-hardware development.

To run the software emulation:

`make sw_emulation`

This basically runs `sdaccel baseline_sw_emulation.tcl`

## Hardware Emulation

The hardware emulation syknthesizes the OpenCL Kernels into RTL and runs co-simulation with the host code. The simulation is run on the host CPU. The HW emulation is slow to build and run (hours for Digit Recongition). This is useful for ensuring that the hardware runs correctly.

To run the hardware emulation:

`make hw_emulation`

## Board Compile

The board compile synthesizes the OpenCL kernels and runs the overall system on the host CPU (host code) and OpenCL FPGA card (kernels). The build process usually takes hours (depending on the size of the design).

To compile and run the board solution:

1. `make board_compile`
2. `source board_setup_env.sh && make board_run`

The arguments for the board compile can be configured in the `Makefile`.
