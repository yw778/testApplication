// ---------------------------------------------------------------------
// Macros for Digit Rec
// ---------------------------------------------------------------------
#define K_CONST 5

#define TRAINING_SIZE 1800
#define TEST_SIZE 2000
#define RESULT_SIZE 2000
#define UNROLL_FACTOR 10

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

void update_knn( long test_inst[4], long train_inst[4], int min_distances[K_CONST] ) 
{
  #pragma HLS inline
  // Compute the difference using XOR

  long diff[4];
  int i;

DIFF_LOOP:  for (i = 0; i < 4; i ++ )
  {
    #pragma HLS unroll  
    diff[i] = test_inst[i] ^ train_inst[i];
  }

  int dist = 0;
  int pos = 15;
  // Count the number of set bits
DIST_LOOP:  for (i = 0; i < 64; ++i ) 
  {
    //#pragma HLS UNROLL
    dist += (diff[0] & 0x1);
    dist += (diff[1] & 0x1);
    dist += (diff[2] & 0x1);
    dist += (diff[3] & 0x1);
    diff[0] = (diff[0]) >> 1;
    diff[1] = (diff[1]) >> 1;
    diff[2] = (diff[2]) >> 1;
    diff[3] = (diff[3]) >> 1;
  }

  int max_dist = 0;
  int max_dist_id = K_CONST+1;

  // Find the max distance
  /*for ( int k = 0; k < K_CONST; ++k ) {
    if ( min_distances[k] > max_dist ) {
      max_dist = min_distances[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if ( dist < max_dist )
    min_distances[max_dist_id] = dist;*/
  for (i = 0; i < K_CONST; i ++ )
  {
    //#pragma HLS unroll
    pos = ((dist < min_distances[i]) && (pos > K_CONST)) ? i : pos;
  }

  for (i = K_CONST ;i > 0; i -- )
  {
    //#pragma HLS unroll
    if(i-1 > pos)
      min_distances[i-1] = min_distances[i-2];
    else if (i-1 == pos)
      min_distances[i-1] = dist;
  }

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
int knn_vote( int knn_set[10][K_CONST] ) {
/*  int min_index = 0;
  int i = 0;
  int k = 0;

  // This array keeps keeps of the occurences
  // of each int in the knn_set

  int score[10];

  // Initialize score array
  for ( i = 0; i < 10; ++i ) {
      score[i] = 0;
  }

  // Find KNNs
  for ( k = 0; k < K_CONST; ++k ) {
    int min_dist = 256;
    int min_dist_id = 10;
    int min_dist_record = K_CONST + 1;

   for ( i = 0; i < 10; ++i ) {
        for ( k = 0; k < K_CONST; ++k ) {
        if ( knn_set[i][k] < min_dist ) {
          min_dist = knn_set[i][k];
          min_dist_id = i;
          min_dist_record = k;
        }
      }
    }

    score[min_dist_id]++;
    // Erase the minimum difference entry once it's recorded
    knn_set[min_dist_id][min_dist_record] = 256;
  }

  // Calculate the maximum score
  int max_score = 0;
  for ( i = 0; i < 10; ++i ) {
    if ( score[i] > max_score ) {
      max_score = score[i];
      min_index = i;
    }
  }

  return min_index;*/
  int min_distance_list[K_CONST];
  #pragma HLS array_partition variable=min_distance_list complete dim=0
  int label_list[K_CONST];
  #pragma HLS array_partition variable=label_list complete dim=0
  int vote_list[10];
  #pragma HLS array_partition variable=vote_list complete dim=0

  int pos = 1000;
  int i, j, r;
  for (i = 0;i < K_CONST; i ++ )
  {
    min_distance_list[i] = 256;
    label_list[i] = 9;
  }

  for (i = 0;i < 10; i ++ )
  {
    vote_list[i] = 5;
  }

  for (i = 0;i < 10; i ++ )
  {
    for (j = 0; j < K_CONST; j ++ )
    {
      pos = 1000;
      for (r = 0; r < K_CONST; r ++ )
      {
        pos = ((knn_set[i][j] < min_distance_list[r]) && (pos > K_CONST)) ? r : pos;
      }
      for (r = K_CONST ;r > 0; r -- )
      {
        if(r-1 > pos)
        {
          min_distance_list[r-1] = min_distance_list[r-2];
          label_list[r-1] = label_list[r-2];
        }
        else if (r-1 == pos)
        {
          min_distance_list[r-1] = knn_set[i][j];
          label_list[r-1] = i;
        }
      }
    }
  }

  for (i = 0;i < K_CONST; i ++ )
  {
    vote_list[label_list[i]] += 1;
  }

  int max_vote;
  max_vote = 0;
  for (i = 0;i < 10; i ++ )
  {
    if(vote_list[i] >= vote_list[max_vote])
    {
      max_vote = i;
    }
  }

  return max_vote;

}

void DigitRec(long * global_training_set, long * global_test_set, int* global_results) {
#pragma HLS INTERFACE m_axi port=global_training_set offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=global_test_set offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=global_results offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=global_training_set bundle=control
#pragma HLS INTERFACE s_axilite port=global_test_set bundle=control
#pragma HLS INTERFACE s_axilite port=global_results bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // This array stores K minimum distances per training set
  int knn_set[10][K_CONST];
  #pragma HLS array_partition variable=knn_set complete dim=0

  int i = 0;
  int j = 0;

  long training_set [TRAINING_SIZE][40] ;
  #pragma HLS array_partition variable=training_set complete dim=2
  //#pragma HLS array_partition variable=training_set complete dim=3

  long test_set    [TEST_SIZE][4];
  #pragma HLS array_partition variable=test_set complete dim=2
  int results      [RESULT_SIZE];

  for (i = 0; i < 10; i ++ )
  {
    for (j = 0; j < TRAINING_SIZE ; j ++ )
    {
      training_set[j][i*4+0] = global_training_set[4*i*TRAINING_SIZE+4*j + 0];
      training_set[j][i*4+1] = global_training_set[4*i*TRAINING_SIZE+4*j + 1];
      training_set[j][i*4+2] = global_training_set[4*i*TRAINING_SIZE+4*j + 2];
      training_set[j][i*4+3] = global_training_set[4*i*TRAINING_SIZE+4*j + 3];
    }
  }

  for ( i = 0; i < TEST_SIZE; ++i) 
  {
    test_set[i][0] = global_test_set[i*4 + 0];
    test_set[i][1] = global_test_set[i*4 + 1];
    test_set[i][2] = global_test_set[i*4 + 2];
    test_set[i][3] = global_test_set[i*4 + 3];
  }

 for (int t = 0; t < TEST_SIZE; ++t) {
 
    // Initialize the knn set
    for ( i = 0; i < 10 ; ++i ) 
    {
      for (j = 0; j < K_CONST; j ++ )
    
      // Note that the max distance is 256
      knn_set[i][j] = 256;
    }

TRAINING_LOOP:    for ( i = 0; i < TRAINING_SIZE; ++i ) {
    #pragma HLS pipeline
DIGIT_UNROLL:      for ( j = 0; j < 10; j++ ) {
      #pragma HLS UNROLL factor=10

        // Read a new instance from the training set
        long training_instance[4];
        training_instance[0]  = training_set[i][j*4+0];
        training_instance[1]  = training_set[i][j*4+1];
        training_instance[2]  = training_set[i][j*4+2];
        training_instance[3]  = training_set[i][j*4+3];
        // Update the KNN set
        update_knn( test_set[t], training_instance, knn_set[j] );
      }
    }

    // Compute the final output
    results[t] = knn_vote(knn_set);
  }

  for ( i = 0; i < RESULT_SIZE; ++i) {
      global_results[i] = results[i];
  }

  return;

}

