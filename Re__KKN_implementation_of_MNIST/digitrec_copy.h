//===========================================================================
// digitrec.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef DIGITREC_H
#define DIGITREC_H

#include "typedefs_copy.h"


// The K_CONST value: number of nearest neighbors
#ifndef K_CONST
#define K_CONST 3
#endif

// Top function for digit recognition
int digitrec(char* trainLabels, unsigned char** trainImages,int trainingSize, unsigned char* input);

void replace(int element, int data,struct node array[K_CONST]);

int calc_dist (char* data);

int knn_vote(struct node knn_data [K_CONST]);

int reverseInt (int i);

unsigned char** readImage(char* data);

int labelSize(char* data);

char* readLabels(char* data);

char* readfiles(const char* filename);

#endif
