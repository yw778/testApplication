# SDAccel Environment Setup File
# Edit this file to match local installation of SDAccel
XILINX_SDACCEL=/opt/Xilinx/SDAccel/2015.1

# LD_LIBRARY_PATH setting, replace sdaccel installation with the location of SDAccel in the local system
export LD_LIBRARY_PATH=$XILINX_SDACCEL/runtime/lib/x86_64/:$LD_LIBRARY_PATH

# LD_LIBRARY_PATH setting, replace sdaccel installation with the location of SDAccel in the local system
export LD_LIBRARY_PATH=$XILINX_SDACCEL/lib/lnx64.o/:$LD_LIBRARY_PATH

# The XILINX_OPENCL environment variable needs to point to the root location of the installation.
# Replace sdaccel location root with the root location of the local installation
export XILINX_OPENCL=$XILINX_SDACCEL

# Set the target platform for the application
export XCL_PLATFORM=xilinx_adm-pcie-7v3_1ddr_1_0
