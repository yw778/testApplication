#include <stdlib.h>
#include <sstream>

#include "utils/spamfilter_fileio.hpp"
#include "utils/spamfilter_defs.h"
#include "utils/spamfilter_utils.hpp"

#include "cpp/sgd_baseline.h"
#include "cpp/batch_baseline.h"
#include "cpp/minibatch_baseline.h"
#include "mbgd_1.h"
#include "mbgd_2.h"
//#include "mbgd_spv.h"
#include "sgd_cublas.h"
#include "sgd_single_point.h"

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

    for (size_t threads_per_datapoint = 128;
        threads_per_datapoint <= 512;
        threads_per_datapoint*=2) {

        training_options.config_params["threads_per_datapoint"]
        = threads_per_datapoint;

        for (size_t datapoints_per_block = 1;
            datapoints_per_block <= 8;
            datapoints_per_block*=2) {

            training_options.config_params["datapoints_per_block"]
            = datapoints_per_block;

            trainAndTest(
                trainParallelStochasticGradientDescent2,
                "CUDA SGD",
                data_set,
                training_options,
                benchmark_options);
        }
    }

    size_t batch_sizes[10] = {1, 2, 4, 10, 20, 30, 45, 50, 60, 100};
    for (size_t threads_per_datapoint = 128;
        threads_per_datapoint <= 512;
        threads_per_datapoint*=2) {

        training_options.config_params["threads_per_datapoint"]
        = threads_per_datapoint;

        for (size_t i = 0;
            i < 2;
            i++) {

            training_options.config_params["batch_size"]
            = batch_sizes[i];

            trainAndTest(
                trainParallelMiniBatchGradientDescent,
                "CUDA MBGD1",
                data_set,
                training_options,
                benchmark_options);
        }
    }

    for (size_t threads_per_mini_batch = 128;
        threads_per_mini_batch <= 512;
        threads_per_mini_batch*=2) {

        training_options.config_params["threads_per_mini_batch"]
        = threads_per_mini_batch;

        for (size_t i = 0;
            i < 2;
            i++) {

            training_options.config_params["batch_size"]
            = batch_sizes[i];

            trainAndTest(
                trainParallelMiniBatchGradientDescent2,
                "CUDA MBGD2",
                data_set,
                training_options,
                benchmark_options);

        }
    }

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

    convergenceRate(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);

    convergenceRate(
        trainMiniBatchGradientDescent,
        "MBGD",
        data_set,
        training_options,
        benchmark_options);


    for (size_t threads_per_datapoint = 128;
        threads_per_datapoint <= 512;
        threads_per_datapoint*=2) {

        training_options.config_params["threads_per_datapoint"]
        = threads_per_datapoint;

        for (size_t datapoints_per_block = 1;
            datapoints_per_block <= 8;
            datapoints_per_block*=2) {

            training_options.config_params["datapoints_per_block"]
            = datapoints_per_block;

            convergenceRate(
                trainParallelStochasticGradientDescent2,
                "CUDA SGD",
                data_set,
                training_options,
                benchmark_options);
        }
    }

    size_t batch_sizes[10] = {1, 2, 4, 10, 20, 30, 45, 50, 60, 100};
    for (size_t threads_per_datapoint = 128;
        threads_per_datapoint <= 512;
        threads_per_datapoint*=2) {

        training_options.config_params["threads_per_datapoint"]
        = threads_per_datapoint;

        for (size_t i = 0;
            i < 2;
            i++) {

            training_options.config_params["batch_size"]
            = batch_sizes[i];

            convergenceRate(
                trainParallelMiniBatchGradientDescent,
                "CUDA MBGD1",
                data_set,
                training_options,
                benchmark_options);
        }
    }

    for (size_t threads_per_mini_batch = 128;
            threads_per_mini_batch <= 512;
            threads_per_mini_batch*=2) {

        training_options.config_params["threads_per_mini_batch"]
        = threads_per_mini_batch;

        for (size_t i = 0;
            i < 2;
            i++) {

            training_options.config_params["batch_size"]
            = batch_sizes[i];

            convergenceRate(
                trainParallelMiniBatchGradientDescent2,
                "CUDA MBGD2",
                data_set,
                training_options,
                benchmark_options);

        }
    }
}

void runConvergenceTime(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    convergenceTime(
        trainStochasticGradientDescent1,
        "sgd_cublas",
        data_set,
        training_options,
        benchmark_options);

    // convergenceTime(
    //     trainStochasticGradientDescent,
    //     "SGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    // convergenceTime(
    //     trainBatchGradientDescent,
    //     "BGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    // convergenceTime(
    //     trainMiniBatchGradientDescent,
    //     "MBGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);


    // for (size_t threads_per_datapoint = 128;
    //     threads_per_datapoint <= 512;
    //     threads_per_datapoint*=2) {

    //     training_options.config_params["threads_per_datapoint"]
    //     = threads_per_datapoint;

    //     for (size_t datapoints_per_block = 1;
    //         datapoints_per_block <= 8;
    //         datapoints_per_block*=2) {

    //         training_options.config_params["datapoints_per_block"]
    //         = datapoints_per_block;

    //         convergenceTime(
    //             trainParallelStochasticGradientDescent2,
    //             "CUDA SGD",
    //             data_set,
    //             training_options,
    //             benchmark_options);
    //     }
    // }

    // size_t batch_sizes[12] = {1, 2, 4, 5, 6, 10, 20, 30, 45, 50, 60, 100};
    // for (size_t threads_per_datapoint = 64;
    //     threads_per_datapoint <= 512;
    //     threads_per_datapoint*=2) {

    //     training_options.config_params["threads_per_datapoint"]
    //     = threads_per_datapoint;

    //     for (size_t i = 0;
    //         i < 2;
    //         i++) {

    //         training_options.config_params["batch_size"]
    //         = batch_sizes[i];

    //         convergenceTime(
    //             trainParallelMiniBatchGradientDescent,
    //             "CUDA MBGD1",
    //             data_set,
    //             training_options,
    //             benchmark_options);
    //     }
    // }

    // for (size_t threads_per_datapoint = 128;
    //     threads_per_datapoint <= 512;
    //     threads_per_datapoint*=2) {

    //     training_options.config_params["threads_per_datapoint"]
    //     = threads_per_datapoint;

    //     for (size_t i = 0;
    //         i < 2;
    //         i++) {
    //
    //         training_options.config_params["batch_size"]
    //         = batch_sizes[i];

    //         convergenceTime(
    //             trainParallelMiniBatchGradientDescentSPV,
    //             "CUDA MBGDSPV",
    //             data_set,
    //             training_options,
    //             benchmark_options);

    //     }
    // }

    // for (size_t threads_per_mini_batch = 64;
    //         threads_per_mini_batch <= 1024;
    //         threads_per_mini_batch*=2) {

    //     training_options.config_params["threads_per_mini_batch"]
    //     = threads_per_mini_batch;

    //     for (size_t i = 0;
    //         i < 3;
    //         i++) {

    //         training_options.config_params["batch_size"]
    //         = batch_sizes[i];

    //         convergenceTime(
    //             trainParallelMiniBatchGradientDescent2,
    //             "CUDA MBGD2",
    //             data_set,
    //             training_options,
    //             benchmark_options);

    //     }
    // }
}


int main(int argc, char *argv[]) {

    // Initialize options to default values
    double* step_size = new double;
    TrainingOptions training_options = initDefaultTrainingOptions(step_size);
    BenchmarkOptions benchmark_options = initDefaultBenchmarkOptions();
    std::string path_to_data("data");

    // Initialize factor for number of times to multiply the dataset features
    int* factor = new int;
    *factor = 1;

    // Parse arguments to adjust options
    parse_command_line_args(argc,
        argv,
        factor,
        &training_options,
        &benchmark_options,
        &path_to_data);

    // Model variables
    FeatureType* data_points = new FeatureType[DATA_SET_SIZE * *factor];
    LabelType* labels = new LabelType[NUM_SAMPLES];
    FeatureType* parameter_vector = new FeatureType[NUM_FEATURES * *factor];

    // Read data from files and insert into variables
    std::string str_points_filepath = path_to_data
            + std::string("/shuffledfeats.dat");
    const char* points_filepath = str_points_filepath.c_str();

    std::string str_labels_filepath = path_to_data
                + std::string("/shuffledlabels.dat");
    const char* labels_filepath = str_labels_filepath.c_str();

    if( readData(DATA_SET_SIZE * *factor, points_filepath, data_points, *factor)
        != DATA_SET_SIZE * *factor)
            return EXIT_FAILURE;

    if( readData(NUM_SAMPLES, labels_filepath, labels, 1)
        != NUM_SAMPLES)
            return EXIT_FAILURE;


    // Initialize data set and options structs
    DataSet data_set;
    data_set.data_points = data_points;
    data_set.labels = labels;
    data_set.parameter_vector = parameter_vector;
    data_set.num_data_points = NUM_SAMPLES;
    data_set.num_features = NUM_FEATURES * *factor;

    // Initial shuffle of the data set to mix spam with ham
    // shuffleKeyValue(data_points, labels, NUM_SAMPLES, NUM_FEATURES);

    // runConvergenceRate(data_set, training_options, benchmark_options);
    // runTrainAndTest(data_set, training_options, benchmark_options);
    runConvergenceTime(data_set, training_options, benchmark_options);

    // Free memory and exit
    delete[] data_points;
    delete[] labels;
    delete[] parameter_vector;
    delete step_size;
    delete factor;

    return 0;
}
