# SDAccel command script
# Design = digit recognition (KNN) example

# Define a solution name
create_solution -name board_compile_solution -dir . -force

# Source project setup
source scripts/project_setup.tcl

# Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary test]

# Generate the system estimate report
# report_estimate

# Run the design in CPU emulation mode
run_emulation -flow cpu -args "-d acc -k DigitRec -f test.xclbin"

# Build application for hardware
build_system

# Package the results for the card
package_system

# Run packaged system
run_system -args "-d acc -k DigitRec -f test.xclbin"

