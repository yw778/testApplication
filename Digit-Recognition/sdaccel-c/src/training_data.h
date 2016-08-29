//===========================================================================
// training_data.h
//===========================================================================
// @brief: This header defines a 2D array that includes 10 training sets, 
//         where each set contains 1800 training data.


#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H


const long long training_data[18000 * 4] = {
    #include "../196data/training_set_0.dat" 
    #include "../196data/training_set_1.dat" 
    #include "../196data/training_set_2.dat" 
    #include "../196data/training_set_3.dat" 
    #include "../196data/training_set_4.dat" 
    #include "../196data/training_set_5.dat" 
    #include "../196data/training_set_6.dat" 
    #include "../196data/training_set_7.dat" 
    #include "../196data/training_set_8.dat" 
    #include "../196data/training_set_9.dat"
};

#endif
