# SDAccel command script
# Design = digit recognition (KNN) example

# Define a solution name
open_solution board_compile_solution_backup/board_compile_solution.spr


run_system -args "-d acc -k DigitRec -f test.xclbin"
