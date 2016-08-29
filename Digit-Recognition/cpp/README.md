# Digit Recognition Tutorial

This tutorial goes over how to write, build and run the software (C/C++) and HLS-C Digit Recognition Application.

## Setup Environment

Based on the environment source one of the setup scripts in the top level directory.

1. `source setup-zhang-01.sh`
2. `source setup-zhang-precision.sh`

## Designing an application

This section is a general overview of how to write an application.

### Main
[main](digitrec_test.cpp) interfaces with the top-level DUT to be instantiated on the FPGA.

## Software

The software emulation runs the hardware DUT on the host CPU. This is useful
for functional verification. The design is complied using gcc/g++ and uses a
pure software flow.

To run the software emulation:

`make sw`

## Hardware Emulation

The hardware emulation synthesizes the hardware DUT into RTL and runs
co-simulation with the host code. The simulation is run on the host CPU. The HW
emulation is slow to build and run. This is useful for ensuring that the
hardware runs correctly. The C to RTL flow is achieved using Vivado HLS.

To run the hardware emulation:

`make hls`

After running through the HW emulation flow results are compiled and stored in
the `results` directory. This helps quickly analyze various designs (e.g.
varying K for K-NN in the run.tcl script) for accuracy, performance and area
utilization.
