

#ifndef TYPEDEFS_CUDA
#define TYPEDEFS_CUDA


struct dataInfo{
    int magicnum;
    int size;
    int rows;
    int columns;
    unsigned char** imageData;
};

struct labelInfo{
    int magicnum;
    int size;
    char* labelData;
};


struct node {
  int distance;
  int data;
};





#endif
