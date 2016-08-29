#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and runs the digitrec program
#
# @desc: 1. Enter "make" to compile & execute the digitrec program
#        2. Enter "make clean" to clean up the directory (before submission)


# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

VHLS = vivado_hls
HLS_TCL= run.tcl

HEADERS = digitrec.h \
					training_data.h \
					typedefs.h


SOURCES = digitrec_test.cpp     \
					digitrec.cpp \

ALL_DEPS = $(HEADERS) $(SOURCES)

# Specify compilation flags
CFLAGS = -g -I${VHLS_INC} -DK_CONST=3

# Compile & run SW based digitrec implementation
sw: $(ALL_DEPS)
	@echo "Compiling & executing the digit recogintion program ..."
	g++ ${CFLAGS} $(SOURCES) -o digitrec
	./digitrec

# Compile & run HLS based digitrec implementation
hls: $(HLS_TCL) $(ALL_DEPS)
	$(VHLS) $(HLS_TCL)

# Clean up the directory before submission
clean:
	@echo "Clean up output files"
	rm -rf digitrec vivado_hls.log *nn.prj result out.dat *~
