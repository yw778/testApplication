# Define the target platform of the application
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:2.1

# Host source files
add_files "../../harness/sdaccel/CLKernel.cpp"
add_files "../../harness/sdaccel/CLMemObj.cpp"
add_files "src/main.cpp"
add_files "src/DigitRecKernel.cpp"

# Header files
#
# Testing Data
add_files "src/testing_data.h"
add_files "196data/test_set.dat"

# Golden Model
add_files "196data/expected.dat"

# Training Data
add_files "src/training_data.h"
add_files "196data/training_set_0.dat"
add_files "196data/training_set_1.dat"
add_files "196data/training_set_2.dat"
add_files "196data/training_set_3.dat"
add_files "196data/training_set_4.dat"
add_files "196data/training_set_5.dat"
add_files "196data/training_set_6.dat"
add_files "196data/training_set_7.dat"
add_files "196data/training_set_8.dat"
add_files "196data/training_set_9.dat"

add_files "../../harness/sdaccel/CLKernel.h"
add_files "../../harness/sdaccel/CLMemObj.h"
add_files "src/DigitRecKernel.h"
set_property file_type "c header files" [get_files "../../harness/sdaccel/CLKernel.h"]
set_property file_type "c header files" [get_files "../../harness/sdaccel/CLMemObj.h"]
set_property file_type "c header files" [get_files "src/DigitRecKernel.h"]

set_property file_type "c header files" [get_files "src/testing_data.h"]
set_property file_type "c header files" [get_files "196data/test_set.dat"]
set_property file_type "c header files" [get_files "196data/expected.dat"]
set_property file_type "c header files" [get_files "src/training_data.h"]
set_property file_type "c header files" [get_files "196data/training_set_0.dat"]
set_property file_type "c header files" [get_files "196data/training_set_1.dat"]
set_property file_type "c header files" [get_files "196data/training_set_2.dat"]
set_property file_type "c header files" [get_files "196data/training_set_3.dat"]
set_property file_type "c header files" [get_files "196data/training_set_4.dat"]
set_property file_type "c header files" [get_files "196data/training_set_5.dat"]
set_property file_type "c header files" [get_files "196data/training_set_6.dat"]
set_property file_type "c header files" [get_files "196data/training_set_7.dat"]
set_property file_type "c header files" [get_files "196data/training_set_8.dat"]
set_property file_type "c header files" [get_files "196data/training_set_9.dat"]

# Kernel definition
create_kernel DigitRec -type clc
add_files -kernel [get_kernels DigitRec] "src/digit_rec_kernel.cl"

# Define binary containers
create_opencl_binary test
set_property region "OCL_REGION_0" [get_opencl_binary test]
create_compute_unit -opencl_binary [get_opencl_binary test] -kernel [get_kernels DigitRec] -name K1
