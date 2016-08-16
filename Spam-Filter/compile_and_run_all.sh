make clean
make all
export OPENBLAS_NUM_THREADS=1
case ":$LD_LIBRARY_PATH:" in
    *:/usr/lib/:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/
esac
case ":$LD_LIBRARY_PATH:" in
    *:/usr/local/cuda-7.0/targets/x86_64-linux/lib:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/targets/x86_64-linux/lib ;;
esac
bin/main
