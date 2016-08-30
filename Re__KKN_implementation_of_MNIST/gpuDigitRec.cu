#include <stdio.h>
#include <stdlib.h>
#include "gpuDigit.cuh"
#include "typedefs.cuh"
//#include <iostream>
//#include <fstream>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void calculations(unsigned char* trainImage, unsigned char* testImage,int* distances) {
  int fullIdx = (blockIdx.x * 980) + threadIdx.x;
  //int bIdx = blockIdx.x;
  int tIdx = threadIdx.x;
  //__global__ int* distance;
  unsigned char* diff;
  printf("%d\n", 100);
  diff[fullIdx] = trainImage[fullIdx] ^ testImage[tIdx%99];
  syncthreads();
  calcDistGPU(diff,distances);
  distances[tIdx] = 100;
  printf("%d\n", 200);
  syncthreads();

}

__device__ void calcDistGPU (unsigned char* data, int* distances)
{
  int i = (blockIdx.x * 980) + threadIdx.x;
    if (threadIdx.x<98) {
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 0] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(98<=threadIdx.x<196){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 1] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(196<=threadIdx.x<295){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 2] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(294<=threadIdx.x<392){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 3] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(392<=threadIdx.x<490){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 4] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(490<=threadIdx.x<588){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 5] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(588<=threadIdx.x<686){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 6] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(686<=threadIdx.x<784){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 7] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(784<=threadIdx.x<882){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 8] += 1;
        }
        data[i] >>=1;
      }
    }
    else if(882<=threadIdx.x<980){
      for(int j = 0; j<8; j++){
        if(data[i]&1 == 1)
        {
          distances[blockIdx.x*10 + 9] += 1;
        }
        data[i] >>=1;
      }
    }
    syncthreads();


}



int digitrec(char* trainLabels, unsigned char* trainImages,int trainingSize, unsigned char* input)
{
  //arrays to keep track of k nearest neighbors
  //int knn_data [K_CONST];
  //int knn_distance [K_CONST];
  struct node knn_data[K_CONST];
  int* diff = (int*)malloc(60000*sizeof(int));

  unsigned char* trainImage;
  unsigned char* testImage;

  int* distances;
  //char* diff = malloc;

  //const int arrayBytes = 784*sizeof(char);

  cudaMalloc( &trainImage, 60000*784/8*sizeof(char));
  cudaMalloc(&testImage, 784/8*sizeof(char));
  cudaMalloc( &distances, 60000*sizeof(int));

  //sets distance array to max
  int a;
  for (a = 0; a < K_CONST; a++)
  {
    //knn_distance [a] = 50;
    //knn_data [a] = 0;
    knn_data[a].data = 0;
    knn_data[a].distance = 785;
  }

  //int i;
  //int j;
  //for (i = 0; i < trainingSize; i++)
  //{
    /*for(j = 0; j < 784; j++)
    {
      diff[j] = trainImages[i][j] ^ input[j];
    }*/

    //printf("Hello\n");

    cudaMemcpy(trainImage, trainImages, 60000*784/8*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(testImage, input, 784/8*sizeof(char), cudaMemcpyHostToDevice);

    //printf("Hello1\n");
    calculations<<<6000, 980>>>(trainImage, testImage, distances);

    //printf("Hello2\n");
    cudaMemcpy(diff, distances, 60000*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Hello3\n");


    for(int i = 0; i<100; i++)
    {
      if(i%28 == 0)
      {
        printf("\n");
      }
      printf("%i", diff[i]);

    }


      //int distance = calc_dist(diff);
      //replace(distances,trainLabels, knn_data);
      //count number of digits that are different.
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


  //}

  free(diff);
  cudaFree(testImage);
  cudaFree(trainImage);
  cudaFree(distances);

  int return_val = knn_vote(knn_data);
  return return_val;
}

void replace(int* element,char* data, struct node array[K_CONST])
{
  for (int i = 0; i< 60000; i++)
  {
    if(array[0].distance > element[i] )
    {
        int j=1;
        while((element[i]<array[j].distance)&&(j<K_CONST))
        {
            array[j-1]=array[j];
            j++;
        }
        array[j-1].distance=element[i];
        array[j-1].data = data[i];
    }
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
  unsigned char** testImages = readImage2d(testing_data);
  char* trainLabels = readLabels(training_labels);
  unsigned char* trainImages = readImage(training_data);

  int trainingSize = labelSize(training_labels);
  int testingSize = labelSize(testing_labels);

  int i;
  int acc = 0;

  /*printf("\n");
  for(int i = 0; i<784; i++)
  {
    if(i%28 == 0)
    {
      printf("\n");
    }
    printf("%i", testImage[i]);
  }

  printf("\n");
*/


  for(i = 0; i < 1; i++)
  {
    int a = digitrec(trainLabels,trainImages,trainingSize,testImages[i]);
    if(a == testLabels[i]){
      acc++;
    }
    //printf("%i\n", acc);
    printf("image# %i, computed: %i ,expected: %i\n", 0,a,testLabels[i]);
  }

  float eff = ((float)acc/(float)testingSize)*100;
  printf("The accuracy is : %f \n", eff );


  free(testImages);
  free(testLabels);

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

int knn_vote( struct node knn_data [K_CONST] )
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

unsigned char* readImage(char* data)
{
  char column[4];
  char row[4];
  char size[4];
  char magic[4];
  unsigned char* images;
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


  images = (unsigned char*)malloc(((amount * rows * columns)/8) * sizeof(char));
  int r;
  for(r = 0; r < ((amount * rows * columns)/8); r++)
  {
    int i;
    for(i = 0; i < 8; i++)
    {
      images[r] <<= 1;
      unsigned char hi = data[8*r+i+16];
      if ((int) hi < 128 )
        {
          images[r] = 0;
        }
      else
        {
          images[r] = 1;
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

unsigned char** readImage2d(char* data)
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
    images[r] = (unsigned char*)malloc(((rows * columns)/8) * sizeof(char));
    for(int i = 0; i < ((rows * columns)/8); i++)
    {
      for(int j = 0; j < 8; j++)
      {
        images[r][i] <<= 1;
        unsigned char hi = data[(8*i)+(r*rows*columns)+j+16];
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
  }

  return images;
}
