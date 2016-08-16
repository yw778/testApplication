make clean
make baseline-only
export OPENBLAS_NUM_THREADS=1
case ":$LD_LIBRARY_PATH:" in
    *:/opt/OpenBLAS/lib/:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/ ;;
esac
bin/main
