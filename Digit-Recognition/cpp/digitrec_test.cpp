//=========================================================================
// testbench.cpp
//=========================================================================
// @brief: testbench for k-nearest-neighbor digit recongnition application

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <vector>

#include "digitrec.h"

#include <time.h>

//using namespace std;

int main() 
{
  // Output file that saves the test bench results
  std::ofstream outfile;
  outfile.open("out.dat");
  
  // Read input file for the testing set
  std::string line;
  std::ifstream myfile ("data/testing_set.dat");

  clock_t start, end, elapsed_time;

  start = clock();


  //digit [180] training_data;
  //int   [180] input_values;
  std::vector<digit> training_data;
  std::vector<int> input_values;
  //
  int data_index = 0;

  if ( myfile.is_open() ) {
    int error = 0;
    int num_test_insts = 0;
    
    while ( std::getline( myfile, line) ) {
      // Read handwritten digit input and expected digit    
      digit input_digit =
          strtoul( line.substr(0, line.find(",")).c_str(), NULL, 16);
      int input_value =
          strtoul(line.substr(line.find(",") + 1,
                              line.length()).c_str(), NULL, 10);

      //training_data[data_index] = input_digit;
      //input_values [data_index] = input_value;
      //data_index++;
      training_data.push_back(input_digit);
      input_values.push_back (input_value);
    }

    data_index = 0;

    while (training_data.size() > 0) {
      
      // Call design under test (DUT)
      int interpreted_digit = digitrec(training_data.back());
      training_data.pop_back();
      
      // Print result messages to console
      num_test_insts++;
      //std::cout << "#" << std::dec << num_test_insts;
      //std::cout << ": \tTestInstance=" << std::hex << input_digit;
      //std::cout << " \tInterpreted=" << interpreted_digit 
      //          << " \tExpected=" << input_value;
      //// Print result messages to file
      //outfile << "#" << std::dec << num_test_insts;
      //outfile << ": \tTestInstance=" << std::hex << input_digit;
      //outfile << " \tInterpreted=" << interpreted_digit 
      //        << " \tExpected=" << input_value;
      
      // Check for any difference between k-NN interpreted digit vs. expected digit
      if ( interpreted_digit != input_values.back()) {
        error++;
      //  std::cout << " \t[Mismatch!]";
      //  outfile << " \t[Mismatch!]";
      }

      input_values.pop_back();

      //std::cout << std::endl;
      //outfile << std::endl;
    }   

    end = clock();

    elapsed_time = (double) (end - start) * 1000 / CLOCKS_PER_SEC;

    printf("Elapsed time = %f ms \n\n", (double) elapsed_time);

    // Report overall error out of all testing instances
    std::cout << "Overall Error Rate = " << std::setprecision(3)
              << ( (double)error / num_test_insts ) * 100
              << "%" << std::endl;
    outfile << "Overall Error Rate = " << std::setprecision(3)
            << ( (double) error / num_test_insts ) * 100 
            << "%" << std::endl;
    
    // Close input file for the testing set
    myfile.close();
    
  }
  else
      std::cout << "Unable to open file for the testing set!" << std::endl; 
  
  // Close output file
  outfile.close();

  return 0;
}
