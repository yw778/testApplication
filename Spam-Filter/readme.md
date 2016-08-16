#Parallelized Logistic Regression for Spam Email Filtering using CUDA

By Gustavo

##Running the code:

In order to run the baseline (serial implementation in C/C++), make sure you have OpenBLAS installed. Change the installation directory of OpenBLAS in the `Makefile` (BLAS_INSTALL_FOLDER) and in `compile_and_run_baseline.sh` to the one you used. The default directory is `/opt/OpenBLAS`.

Then, use the following commands in a terminal standing at the current directory:

```bash
. compile_and_run_baseline.sh
```
Make sure the data (features and labels) are located in ```./data/```.
