#include <stdio.h>
#include <stdlib.h>
#include "digitrec_copy.h"
#include "typedefs_copy.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>



int digitrec(char* trainLabels, unsigned char** trainImages,int trainingSize, unsigned char* input)
{
  //arrays to keep track of k nearest neighbors
  //int knn_data [K_CONST];
  //int knn_distance [K_CONST];
  struct node knn_data[K_CONST];

  char* diff = (char*)malloc(784*sizeof(char));

  //sets distance array to max
  int a;
  for (a = 0; a < K_CONST; a++)
  {
    //knn_distance [a] = 785;
    //knn_data [a] = 0;
    knn_data[a].data = 0;
    knn_data[a].distance = 785;
  }

  int i;
  int j;
  for (i = 0; i < trainingSize; i++)
  {
    for(j = 0; j < 784; j++)
    {
      diff[j] = trainImages[i][j] ^ input[j];
    }

      //count number of digits that are different.
      int distance = calc_dist(diff);

      //find which knn to replace
      /*int k = 0;
      while ((k < K_CONST ) && (distance >= knn_distance[k]) )
      {
        k++;
      }
      if (k < K_CONST )
      {
        knn_distance[k] = distance;
        knn_data[k] = trainLabels[i];
      }*/
      replace(distance,trainLabels[i], knn_data);
      /*for (j = 0; j < K_CONST; j ++)
      {
        printf("%d ", knn_data[j].distance);
      }
      printf("\n");
      sleep(1);*/
  }

  free(diff);

  int return_val = knn_vote(knn_data);
  /*for(int i=0; i<K_CONST;i++)
  {
  printf("%d\n", knn_data[i].data);
  printf("%d\n", knn_data[i].distance);

}*/
  return return_val;
}

void replace(int element,int data, struct node array[K_CONST])
{
  if(array[0].distance > element )
  {
      int j=1;
      while((element<array[j].distance)&&(j<K_CONST))
      {
          array[j-1]=array[j];
          j++;
      }
      array[j-1].distance=element;
      array[j-1].data = data;
  }
}

int main ()
{
  struct timeval t0,t1 ;
  gettimeofday(&t0, 0);

  char* testing_data;
  char* testing_labels;
  char* training_data;
  char* training_labels;
  const char* testDataFile = "bigData/t10k-images-idx3-ubyte";
  const char* testLabelFile = "bigData/t10k-labels-idx1-ubyte";
  const char* trainDataFile = "bigData/train-images-idx3-ubyte";
  const char* trainLabelFile = "bigData/train-labels-idx1-ubyte";

  testing_data = readfiles(testDataFile);
  testing_labels = readfiles(testLabelFile);
  training_data = readfiles(trainDataFile);
  training_labels = readfiles(trainLabelFile);

  char* testLabels = readLabels(testing_labels);
  unsigned char** testImages = readImage(testing_data);
  char* trainLabels = readLabels(training_labels);
  unsigned char** trainImages = readImage(training_data);

  int trainingSize = labelSize(training_labels);
  int testingSize = labelSize(testing_labels);

  /*for(int i = 0; i<784; i++)
  {
    if(i%28 == 0)
    {
      printf("\n");
    }
    printf("%i", testImages[0][i]);

  }*/

  int i;
  int acc = 0;
  for(i = 0; i < testingSize; i++)
  {
    int a = digitrec(trainLabels,trainImages,trainingSize,testImages[i]);
    if(a == testLabels[i]){
      acc++;
    }
    //printf("%i\n", acc);
    //printf("image# %i, computed: %i ,expected: %i\n", 0,a,testLabels[i]);
  }

  float eff = ((float)acc/(float)testingSize)*100;
  printf("The accuracy is : %f \n", eff );

  int j;
  for (j = 0; j< 10000; j++)
  {
    free(testImages[j]);
  }
  free(testImages);
  free(testLabels);

  int b;
  for (b = 0; b< trainingSize; b++)
  {
    free(trainImages[b]);
  }
  free(trainImages);
  free(trainLabels);



  gettimeofday(&t1, 0);
  long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
  double ti = (float)elapsed/1000000;
  printf("Time elasped:%ld microsec, %f sec\n", elapsed , ti);

  return  0;
}


int calc_dist (char* data)
{
  int distance = 0;
  int i;
  for(i = 0; i < 784; i++)
  {
    if (data[i] == 1) {
      distance++;
    }
  }
  return distance;
}

int reverseInt (int i) //reverse a int from high-endian to low-endian
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int knn_vote(struct node knn_data [K_CONST] )
{
  int j;
  int final [10] = {0,0,0,0,0,0,0,0,0,0};
  int i;
  for (i = 0; i < K_CONST; i++)
  {
    j = knn_data[i].data;
    final[j]++;
  }
 int maximum=-1;
 int location =0;

 int c;
  for (c = 0; c < 10; c++)
  {
    if (final[c] > maximum)
    {
       maximum  = final[c];
       location = c;
    }
  }
 return location;

}

unsigned char** readImage(char* data)
{
  char column[4];
  char row[4];
  char size[4];
  char magic[4];
  unsigned char** images;
  memcpy(column, data+12, 4);
  memcpy(row, data+8, 4);
  memcpy(size, data+4, 4);
  memcpy(magic, data, 4);

  int rcolumn =  *((int*)column);
  int columns = reverseInt(rcolumn);
  int rrow =  *((int*)row);
  int rows = reverseInt(rrow);
  int ramount =  *((int*)size);
  int amount = reverseInt(ramount);
  int rmagicnum =  *((int*)magic);
  int magicnum = reverseInt(rmagicnum);


  images = (unsigned char**)malloc(amount * sizeof(char *));
  int r;
  for(r = 0; r < amount; r++)
  {
    images[r] = (unsigned char*)malloc(rows * columns * sizeof(char));
    int i;
    for(int i = 0; i < (rows*columns); i++)
    {
      unsigned char hi = data[i+(r*rows*columns)+16];
      int now = hi -'0';
      if ((int) hi < 128 )
      {
        images[r][i] = 0;
      }
      else
      {
        images[r][i] = 1;
      }
    }
  }

  return images;
}

char* readLabels(char* data)
{
  char size[4];
  char magic[4];
  char* labels;
  memcpy(size, data+4, 4);
  memcpy(magic, data, 4);

  int ramount =  *((int*)size);
  int amount = reverseInt(ramount);
  int rmagicnum =  *((int*)magic);
  int magicnum = reverseInt(rmagicnum);


  labels = (char*)malloc(amount * sizeof(char));
  int i;
  for(i = 0; i < amount; i++)
  {
    labels[i] = data[i+8];
  }

  return labels;
}

int labelSize(char* data)
{
  char size[4];
  char magic[4];
  memcpy(size, data+4, 4);
  memcpy(magic, data, 4);

  int ramount =  *((int*)size);
  int amount = reverseInt(ramount);
  int rmagicnum =  *((int*)magic);
  int magicnum = reverseInt(rmagicnum);


  return amount;
}

char* readfiles(const char* filename)
{
  FILE* fileptr;
  char* data;
  long filelen;

  fileptr = fopen(filename, "rb");  // Open the file in binary mode
  fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
  filelen = ftell(fileptr);           // Get the current byte offset in the file

  rewind(fileptr);                      // Jump back to the beginning of the file
  (data) = (char *)malloc((filelen+1)*sizeof(char)); // Enough memory for file + \0
  fread(data, filelen, 1, fileptr); // Read in the entire file

  fclose(fileptr); // Close the file

  return data;
}
