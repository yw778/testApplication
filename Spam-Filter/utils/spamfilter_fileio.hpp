#ifndef SPAMFILTER_FILEIO
#define SPAMFILTER_FILEIO

#include <fstream>
#include <iostream>
//#include <cstdio>
#include <stdlib.h>

using namespace std;

//read a text file and insert lines in vector
//return number of lines read; 0 if file failed to open
template <typename Type>
size_t readData(size_t size, const char* file_name, Type* vector, Type bias, int num_featrures) {

    std::ifstream source_file;
    source_file.open(file_name);

    if (!source_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return 0;
    }
    size_t i = 0;    
    //use to track insert place
    size_t insert_track = 0;
    for (Type val = 0;  i < size ;  i++) {        
        //insert bias to datapoints every 1024 points
        insert_track++;
        if( insert_track == num_featrures){
            vector[i] = bias;
            insert_track = 0;            
        }else{
            source_file >> val;
            vector[i] = val; 
            // printf("%d data reading\n",i);
        }
    }

    source_file.close();
    cout << "Finished reading " << i << " lines of " << file_name << "\n";
    return i;
}

template <typename Type>
size_t readLabel(size_t size, const char* file_name, Type* vector) {

    std::ifstream source_file;
    source_file.open(file_name);

    if (!source_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return 0;
    }

    size_t i = 0;    
    for (Type val = 0; i < size && source_file >> val; i++) {        
        vector[i] = val;
    }

    source_file.close();
    cout << "Finished reading " << i << " lines of " << file_name << "\n";
    return i;
}


template <typename Type>
size_t readData(size_t size, const char* file_name, Type* vector) {

    std::ifstream source_file;
    source_file.open(file_name);

    if (!source_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return 0;
    }

    size_t i = 0;    
    for (Type val = 0; i < size && source_file >> val; i++) {        
        vector[i] = val;
    }

    source_file.close();
    cout << "Finished reading " << i << " lines of " << file_name << "\n";
    return i;
}

//write content of a vector to a text file, separated by \n
template <typename Type>
void writeData(size_t size, Type* vector, const char* file_name = "output.txt") {

    std::ofstream output_file;
    output_file.open(file_name, ios_base::trunc);

    if (!output_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return;
    }

    size_t i;
    for (i = 0; i < size; i++) {
        output_file << vector[i] << "\n";
    }

    output_file.close();
    cout << "Finished writing " << i << " lines of " << file_name << "\n";
}

#endif
