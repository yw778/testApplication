# Digit Recognition

Digit recognition leverages K-Nearest Neighbor Machine Learning techniques to
automatically categorize handwritten digits as one of 0 through 9. The
handwritten digits come from the MNIST database. After the digits have been
downsampled and undergo thresholding, they are represented as a 7 x 7 grid of
0's and 1's which outline a handwritten digit. The data set is made up of 1800
instances per digit (18,000 total) and the test set is another 180 instances.


The `cpp` directory holds the software and Vivado HLS based hardware design.
This folder provides users with quick software functional correctness and rapid
HW/SW development.

The `sdaccel-c` and `sdaccel-opencl` leverage the SDAccel tool flow for C and
OpenCL based kernels respectively.
