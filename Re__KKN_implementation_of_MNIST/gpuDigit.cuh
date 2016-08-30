#ifndef gpuDigit_CUDA
#define gpuDigit_CUDA

#include "typedefs.cuh"


// The K_CONST value: number of nearest neighbors
#ifndef K_CONST
#define K_CONST 3
#endif

// Top function for digit recognition

__device__ void calcDistGPU (unsigned char* data, int*distances);

int digitrec(char* trainLabels, unsigned char* trainImages,int trainingSize, unsigned char* input);

void replace(int* element,char* data, struct node array[K_CONST]);

int calc_dist (char* data);

int knn_vote(struct node knn_data [K_CONST]);

int reverseInt (int i);

unsigned char* readImage(char* data);

int labelSize(char* data);

char* readLabels(char* data);

char* readfiles(const char* filename);

unsigned char** readImage2d(char* data);


#endif
