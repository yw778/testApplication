#include <stdlib.h>

#include "utils/spamfilter_fileio.hpp"
#include "utils/spamfilter_defs.h"
#include "utils/spamfilter_utils.hpp"

#include "sgd_baseline.h"
#include "batch_baseline.h"
#include "minibatch_baseline.h"

void runTrainAndTest(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    trainAndTest(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    trainAndTest(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);

    trainAndTest(
        trainMiniBatchGradientDescent,
        "MBGD",
        data_set,
        training_options,
        benchmark_options);

}

void runConvergenceRate(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    convergenceRate(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    // convergenceRate(
    //     trainBatchGradientDescent,
    //     "BGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    // convergenceRate(
    //     trainMiniBatchGradientDescent,
    //     "MBGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

}

void runConvergenceTime(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    convergenceTime(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    convergenceTime(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);
    
    size_t batch_sizes[10] = {1, 2, 4, 10, 20, 30, 45, 50, 60, 100};
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes[i];
        convergenceTime(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

}

int main(int argc, char *argv[]) {

    // Initialize options to default values
    double* step_size = new double;
    TrainingOptions training_options = initDefaultTrainingOptions(step_size);
    BenchmarkOptions benchmark_options = initDefaultBenchmarkOptions();
    std::string path_to_data("data");

    // Parse arguments to adjust options
    parse_command_line_args(argc,
        argv,
        &training_options,
        &benchmark_options,
        &path_to_data);

    // Model variables
    FeatureType* data_points = new FeatureType[DATA_SET_SIZE];
    LabelType* labels = new LabelType[NUM_SAMPLES];
    FeatureType* parameter_vector = new FeatureType[NUM_FEATURES];

    // Read data from files and insert into variables
    std::string str_points_filepath(path_to_data
                                    + std::string("/shuffledfeats.dat"));

    std::string str_labels_filepath(path_to_data
                                    + std::string("/shuffledlabels.dat"));

    const char* labels_filepath = str_labels_filepath.c_str();
    const char* points_filepath = str_points_filepath.c_str();

    std::cout<< benchmark_options.bias<< endl;

    if( readData(DATA_SET_SIZE, points_filepath, data_points, benchmark_options.bias, NUM_FEATURES)
        != DATA_SET_SIZE)
            return EXIT_FAILURE;
    if( readLabel(NUM_SAMPLES, labels_filepath, labels)
        != NUM_SAMPLES)
            return EXIT_FAILURE;

    // Initialize data set and options structs
    DataSet data_set;
    data_set.data_points = data_points;
    data_set.labels = labels;
    data_set.parameter_vector = parameter_vector;
    data_set.num_data_points = NUM_SAMPLES;
    data_set.num_features = NUM_FEATURES;
    // Initial shuffle of the data set to mix spam with ham
    // shuffleKeyValue(data_points, labels, NUM_SAMPLES, NUM_FEATURES);

     runConvergenceRate(data_set, training_options, benchmark_options);
    // runTrainAndTest(data_set, training_options, benchmark_options);
    // runConvergenceTime(data_set, training_options, benchmark_options);

    // Free memory and exit
    delete step_size;
    delete[] data_points;
    delete[] labels;
    delete[] parameter_vector;

    return 0;
}
