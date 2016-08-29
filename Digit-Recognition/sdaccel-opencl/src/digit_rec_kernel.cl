// ---------------------------------------------------------------------
// Macros for OpenCL Optimization
// ---------------------------------------------------------------------
#define LOOP_PIPELINE __attribute__((xcl_pipeline_loop))
#define LOOP_UNROLL __attribute__((opencl_unroll_hint))

// ---------------------------------------------------------------------
// Macros for Digit Rec
// ---------------------------------------------------------------------
#define K_CONST 1

#define TRAINING_SIZE 1800
#define TEST_SIZE 2000
#define RESULT_SIZE 2000
#define UNROLL_FACTOR 10

typedef struct tag_long4_t 
{
  long x;
  long y;
  long z;
  long w;
} long4_t;


// #include "training_data.h" 
//-----------------------------------------------------------------------
// update_knn function
//-----------------------------------------------------------------------
// Given the test instance and a (new) training instance, this
// function maintains/updates an array of K minimum
// distances per training set.

// @param[in] : test_inst - the testing instance
// @param[in] : train_inst - the training instance
// @param[in/out] : min_distances[K_CONST] - the array that stores the current
//                  K_CONST minimum distance values per training set

void update_knn( long4 test_inst, long4 train_inst, int min_distances[K_CONST] ) {
  // Compute the difference using XOR
  long4 diff = test_inst ^ train_inst;

  int dist = 0;
  // Count the number of set bits
  LOOP_UNROLL for ( int i = 0; i < 64; ++i ) {
    dist += (diff.x & 0x1);
    dist += (diff.y & 0x1);
    dist += (diff.z & 0x1);
    dist += (diff.w & 0x1);
    diff.x = diff.x >> 1;
    diff.y = diff.y >> 1;
    diff.z = diff.z >> 1;
    diff.w = diff.w >> 1;
  }



  int max_dist = 0;
  int max_dist_id = K_CONST+1;
  int k = 0;

  // Find the max distance
  for ( int k = 0; k < K_CONST; ++k ) {
    if ( min_distances[k] > max_dist ) {
      max_dist = min_distances[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if ( dist < max_dist )
    min_distances[max_dist_id] = dist;

  return;
}


//-----------------------------------------------------------------------
// knn_vote function
//-----------------------------------------------------------------------
// Given 10xK minimum distance values, this function
// finds the actual K nearest neighbors and determines the
// final output based on the most common int represented by
// these nearest neighbors (i.e., a vote among KNNs).
//
// @param[in] : knn_set - 10xK_CONST min distance values
// @return : the recognized int
__local int knn_vote( int knn_set[10 * K_CONST] ) 
{
  int min_distance_list[K_CONST] __attribute__((xcl_array_partition(complete, 1)));
  int label_list[K_CONST]  __attribute__((xcl_array_partition(complete, 1)));
  int vote_list[10]  __attribute__((xcl_array_partition(complete, 1)));

  int pos = 1000;

  /*LOOP_UNROLL*/ for (int i = 0;i < K_CONST; i ++ )
  {
    min_distance_list[i] = 256;
    label_list[i] = 9;
  }

  LOOP_UNROLL for (int i = 0;i < 10; i ++ )
  {
    vote_list[i] = 5;
  }

  for (int i = 0;i < 10; i ++ )
  {
    /*LOOP_UNROLL*/ for (int j = 0; j < K_CONST; j ++ )
    {
      pos = 1000;
      for (int r = 0; r < K_CONST; r ++ )
      {
        pos = ((knn_set[i*K_CONST+j] < min_distance_list[r]) && (pos > K_CONST)) ? r : pos;
      }
      for (int r = K_CONST ;r > 0; r -- )
      {
        if(r-1 > pos)
        {
          min_distance_list[r-1] = min_distance_list[r-2];
          label_list[r-1] = label_list[r-2];
        }
        else if (r-1 == pos)
        {
          min_distance_list[r-1] = knn_set[i*K_CONST+j];
          label_list[r-1] = i;
        }
      }
    }
  }

  /*LOOP_UNROLL*/ for (int i = 0;i < K_CONST; i ++ )
  {
    vote_list[label_list[i]] += 1;
  }

  __local int max_vote;
  max_vote = 0;
  LOOP_UNROLL for (int i = 0;i < 10; i ++ )
  {
    if(vote_list[i] >= vote_list[max_vote])
    {
      max_vote = i;
    }
  }

  return max_vote;

}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
//__kernel void DigitRec(__global long long * global_training_set, __global long long * global_test_set, __global long long * global_results) {
__kernel void DigitRec(__global long4 * global_training_set, __global long4 * global_test_set, __global int * global_results) {
  //printf("Digitrec started!\n");
  event_t results_copy;
  event_t data_copy;

  // This array stores K minimum distances per training set
  int knn_set[10 * K_CONST] __attribute__((xcl_array_partition(complete, 1)));

  __local long training_set  [TRAINING_SIZE][40] __attribute__((xcl_array_partition(cyclic, 10, 2)));
  __local long4 test_set     [TEST_SIZE];
  __local int results        [RESULT_SIZE];


  for (int i = 0; i < 10; i ++ )
  {
    for (int j = 0; j < TRAINING_SIZE ; j ++ )
    {
      long4 tmp = global_training_set[i*TRAINING_SIZE+j];
      training_set[j][i*4  ] = tmp.x;
      training_set[j][i*4+1] = tmp.y;
      training_set[j][i*4+2] = tmp.z;
      training_set[j][i*4+3] = tmp.w;
    }
  }
  data_copy = async_work_group_copy(test_set     , global_test_set     , TEST_SIZE         , 0);

  wait_group_events(1, &data_copy);

  for (int t = 0; t < TEST_SIZE; ++t) 
  {

    long4 test_instance = test_set[t];

    // Initialize the knn set
    LOOP_UNROLL for ( int i = 0; i < K_CONST * 10 ; ++i ) {
      // Note that the max distance is 256
      knn_set[i] = 256;
    }
    TRAINING_LOOP : LOOP_PIPELINE for ( int i = 0; i < TRAINING_SIZE; ++i ) {
    LANES : LOOP_UNROLL for ( int j = 0; j < 10; j++ ) {

        // Read a new instance from the training set
        long4 training_instance;
        training_instance.x= training_set[i][j * 4    ];
        training_instance.y= training_set[i][j * 4 + 1];
        training_instance.z= training_set[i][j * 4 + 2];
        training_instance.w= training_set[i][j * 4 + 3];

        // Update the KNN set
        update_knn( test_instance, training_instance, &knn_set[j * K_CONST] );
      }
    }
    // Compute the final output
    results[t] = knn_vote(knn_set);
  }

  results_copy = async_work_group_copy(global_results, results, RESULT_SIZE, 0);

  wait_group_events(1, &results_copy);
}

