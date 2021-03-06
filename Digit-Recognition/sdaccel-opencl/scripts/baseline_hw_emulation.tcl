# SDAccel command script
# Design = digit recognition (KNN) example

# Define a solution name
create_solution -name baseline_hw_solution -dir . -force

# Source project setup
source scripts/project_setup.tcl

# Compile the design for CPU based emulation
compile_emulation -flow hardware -opencl_binary [get_opencl_binary test]

# Generate the system estimate report
report_estimate

# Run the design in CPU emulation mode
run_emulation -flow hardware -args "-d acc -k DigitRec -f test.xclbin"
