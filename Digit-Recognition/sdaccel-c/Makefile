CC = sdaccel

HARNESS_DIR = ../../harness/sdaccel

SW_EMUL_TCL = scripts/baseline_sw_emulation.tcl
HW_EMUL_TCL = scripts/baseline_hw_emulation.tcl
DEBUG_TCL   = scripts/debug.tcl
BOARD_COMPILE_TCL = scripts/board_compile.tcl
BOARD_RUN_TCL := scripts/board_run.tcl

HEADERS = $(HARNESS_DIR)/CLKernel.h \
					$(HARNESS_DIR)/CLMemObj.h \
					src/DigitRecKernel.h

SOURCES = src/main.cpp     \
					$(HARNESS_DIR)/CLKernel.cpp \
					$(HARNESS_DIR)/CLMemObj.cpp \
					src/DigitRecKernel.cpp

ALL_DEPS = $(HEADERS) $(SOURCES)

sw_emulation: $(SW_EMUL_TCL) $(ALL_DEPS)
	$(CC) $(SW_EMUL_TCL)

hw_emulation: $(HW_EMUL_TCL) $(ALL_DEPS)
	$(CC) $(HW_EMUL_TCL)

board_compile: $(BOARD_COMPILE_TCL) $(ALL_DEPS)
	$(CC) $(BOARD_COMPILE_TCL)

board_run:
	cd board_compile_solution/pkg/pcie && ./board_compile_solution.exe -d acc -k DigitRec -f test.xclbin

debug: $(DEBUG_TCL) $(ALL_DEPS)
	$(CC) $(DEBUG_TCL)

clean:
	rm -f *.jou
	rm -f *.log
	rm -rf .Xil/
	rm -rf baseline_sw_solution/
	rm -rf baseline_hw_solution/
	rm -rf debug_solution/
