#include <math.h>
#include <getopt.h>
#include <string.h>

#include "spamfilter_utils.hpp"
#include "spamfilter_timer.h"



/******************************************************************************/
/*                                                                            */
/* General Utilities                                                          */
/*                                                                            */
/******************************************************************************/

// initialize default configuration options
TrainingOptions initDefaultTrainingOptions(double* step_size) {

    Dictionary default_config_params;
    default_config_params["tolerance"] = TOLERANCE;
    default_config_params["initial_step_size"] = STEP_SIZE;

    (*step_size) = STEP_SIZE;

    TrainingOptions default_training_options =
        {NUM_EPOCHS, step_size, default_config_params};

    return default_training_options;
}


BenchmarkOptions initDefaultBenchmarkOptions() {

    static const BenchmarkOptions default_benchmark_options =
        {NUM_TRAINING, NUM_TESTING, MAX_NUM_EPOCHS, NUM_RUNS, ERROR_GOAL, 0.0};

    return default_benchmark_options;
}


// prints usage in case an invalid command line argument is passed
static void print_usage()
{
    printf("usage: <name_of_binary> <options>\n");
    printf("options:\n");
    printf("    -e num_epochs\n");
    printf("    -z step_size\n");
    printf("    -t num_points_training\n");
    printf("    -s num_points_testing\n");
    printf("    -m max_num_epochs\n");
    printf("    -r num_runs\n");
    printf("    -g error_goal\n");
    printf("    -p path_to_data_directory\n");
    printf("    -b [biasScalingFactor]\n");
}

// parses training- and benchmark- related command line arguments
void parse_command_line_args(
    int argc,
    char ** argv,
    TrainingOptions* training_options,
    BenchmarkOptions* benchmark_options,
    std::string* path_to_data) {

    int c = 0;

    while ((c = getopt(argc, argv, "e:z:t:s:m:r:g:p:b:")) != -1) {
        switch (c) {
            case 'e':
                if (training_options)
                    training_options->num_epochs = atol(optarg);
                break;
            case 'z':
                if (training_options)
                    *(training_options->step_size)
                        = training_options->config_params["initial_step_size"]
                        = atof(optarg);
                break;
            case 't':
                if (benchmark_options)
                    benchmark_options->num_points_training = atol(optarg);
                break;
            case 's':
                if (benchmark_options)
                    benchmark_options->num_points_testing = atol(optarg);
                break;
            case 'm':
                if (benchmark_options)
                    benchmark_options->max_num_epochs = atol(optarg);
                break;
            case 'r':
                if (benchmark_options)
                    benchmark_options->num_runs = atoi(optarg);
                break;
            case 'g':
                if (benchmark_options)
                    benchmark_options->error_goal = atof(optarg);
                break;
            case 'p':
                if (path_to_data)
                    *path_to_data = optarg;
                break;
            case 'b':
                if (benchmark_options)
                    benchmark_options->bias = atof(optarg);
                break;
            default:
                print_usage();
        } // matching on arguments
    } // while args present
}




/******************************************************************************/
/*                                                                            */
/* Logistic Regression-Specific Utilities                                     */
/*                                                                            */
/******************************************************************************/

// computes logistic function for a given double
double logisticFunction(double exponent) {
    return (1.0 / (1.0 + exp(-exponent)));
}

// computes logistic function for a given parameter vector (parameter_vector)
// and a data point (data_point_i)
double logisticFunction(
    FeatureType* parameter_vector,
    FeatureType* data_point_i,
    const size_t num_features) {

    return logisticFunction(
        dotProduct(parameter_vector, data_point_i, num_features));
}


// gets predicted label for a given parameter vector (parameter_vector)
// and a given datapoint (data_point_i)
LabelType getPrediction(
    FeatureType* parameter_vector,
    FeatureType* data_point_i,
    size_t num_features,
    const double treshold = 0) {

    float parameter_vector_dot_x_i =
        dotProduct(parameter_vector, data_point_i, num_features);

    return
        (parameter_vector_dot_x_i > treshold)? POSITIVE_LABEL : NEGATIVE_LABEL;
}


// serial implementation of accuracy test
// computes true positive rate, false positive rate and error rate, then adds
// these values to variables passed through pointers
double computeErrorRate(
    DataSet data_set,
    Dictionary* errors){

    size_t true_positives = 0, true_negatives = 0,
        false_positives = 0, false_negatives = 0;

    // for each point in the given data_set, compare prediction with actual label
    for (size_t i = 0; i < data_set.num_data_points; i++) {

        LabelType prediction = getPrediction(
            data_set.parameter_vector,
            &data_set.data_points[i * data_set.num_features],
            data_set.num_features);

        if (prediction != data_set.labels[i]){
            if (prediction == POSITIVE_LABEL)
                false_positives++;
            else
                false_negatives++;
        } else {
            if (prediction == POSITIVE_LABEL)
                true_positives++;
            else
                true_negatives++;
        }
    }

    double error_rate =
        (double) (false_positives + false_negatives) / data_set.num_data_points;

    // if a pointer to a Dictionary of errors was passed, accumulate the results
    if (errors) {
        (*errors)["tpr"]
            += (double)true_positives / (true_positives + false_negatives);
        (*errors)["fpr"]
            += (double)false_positives / (false_positives + true_negatives);
        (*errors)["error"]
            += error_rate;
    }

    return error_rate;
}

// loss function for the whole batch (Negative LogLikelihood for
// Bernoulli probability distribution)
double lossFunction(DataSet data_set) {

    double sum = 0.0;

    for (size_t i = 0; i < data_set.num_data_points; i++) {

        double probability_of_positive = logisticFunction(
            data_set.parameter_vector,
            &data_set.data_points[i * data_set.num_features],
            data_set.num_features);

        FeatureType label_i = (FeatureType) data_set.labels[i];

        sum += ((label_i * log(probability_of_positive))
                + ((1 - label_i) * log(1 - probability_of_positive)))
                    / data_set.num_data_points;
    }

    return -sum; // negative of the sum
}

// resets parameter vector to "forget" previous training
void resetParameters(
    DataSet data_set,
    TrainingOptions training_options) {

    memset(
        data_set.parameter_vector,
        0,
        data_set.num_features * sizeof(FeatureType));

    (*training_options.step_size) =
        training_options.config_params["initial_step_size"];
}

// splits the data_set into training and testing sets
void splitDataSet(
    DataSet data_set,
    BenchmarkOptions benchmark_options,
    DataSet* training_set,
    DataSet* testing_set) {

    (*training_set) = data_set;
    training_set->num_data_points = min_val(
        benchmark_options.num_points_training,
        data_set.num_data_points);

    (*testing_set) = data_set;
    testing_set->data_points =
        &data_set.data_points[data_set.num_features
                              * benchmark_options.num_points_training];
    testing_set->labels =
        &data_set.labels[benchmark_options.num_points_training];
    testing_set->num_data_points = min_val(
        benchmark_options.num_points_testing,
        data_set.num_data_points - training_set->num_data_points);

}

// determines whether a key already exists in a Dictionary
bool fieldExists(Dictionary dict, const char* key) {
    return (dict.find(key) != dict.end());
}

// initialize error dictionary fields to zero
void initErrorDict(Dictionary* dict) {
    (*dict)["tpr"] = 0.0;
    (*dict)["fpr"] = 0.0;
    (*dict)["error"] = 0.0;
}

// divide cumulative errors by number of runs
void scaleErrorDict(Dictionary* dict, float factor) {
    (*dict)["tpr"] *= factor;
    (*dict)["fpr"] *= factor;
    (*dict)["error"] *= factor;
}

// print CSV header only if it hasn't been printed before
void maybePrintHeader(const char* header) {

    static bool already_printed_header = false;

    if (!already_printed_header && header) {
        printf("%s", header);
        already_printed_header = true;
    }
}

// Makes a printable representation of the list of configuration parameters
static std::string configParamsToStr(Dictionary config_params) {

    std::string result("");

    for (Dictionary::iterator param = config_params.begin();
        param != config_params.end();
        ++param) {

        char second[100];
        sprintf(second, "%g", param->second);
        result += param->first + ":" + second + "&";
    }

    return result.substr(0, result.size()-1);
}

// Benchmark method for the logistic regression model.
// It takes a training function, a data set and some configuration options,
// and then calls the training function several times until the desired
// accuracy is reached or the number of epochs exceeds the given limit.
// The training time is measured for multiple runs and the results are averaged.
void trainAndTest(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    // Initialize benchmark variables
    Timer training_timer(name);
    double training_time = 0.0;

    double average_epochs_spent = 0.0;

    Dictionary train_errors;
    initErrorDict(&train_errors);

    Dictionary test_errors;
    initErrorDict(&test_errors);

    DataSet training_set, testing_set;
    splitDataSet(data_set, benchmark_options, &training_set, &testing_set);

    // Ignore the first run to discard initialization overhead
    training_function(training_set, training_options);

    // Shuffle vectors and train num_runs times, then average results
    for (size_t k = 0; k < benchmark_options.num_runs; k++) {

        // reset parameter vector to forget previous training
        resetParameters(training_set, training_options);

        // shuffleKeyValue(data_points, labels, num_points_total, num_features);
        training_timer.start();
        training_function(training_set, training_options);
        training_time += training_timer.stop();

        // Get Training error
        computeErrorRate(training_set, &train_errors);

        // Get Testing error
        computeErrorRate(testing_set, &test_errors);

    }
    // reset configuration parameters
    *training_options.step_size =
        training_options.config_params["initial_step_size"];

    training_time /= benchmark_options.num_runs;
    scaleErrorDict(&train_errors, 100.0 / benchmark_options.num_runs);
    scaleErrorDict(&test_errors, 100.0 / benchmark_options.num_runs);

    // Output results
    maybePrintHeader("name,config params,runs,epochs,train time,train TPR,"
                     "train FPR,train Error,test TPR,test FPR,test Error\n");

    printf( "%s,%s,%lu,%lu,%f,%f,%f,%f,%f,%f,%f\n",
            name,
            configParamsToStr(training_options.config_params).c_str(),
            benchmark_options.num_runs,
            training_options.num_epochs,
            training_time,
            train_errors["tpr"],
            train_errors["fpr"],
            train_errors["error"],
            test_errors["tpr"],
            test_errors["fpr"],
            test_errors["error"]);
}

// prints error rates for a given set of epochs
void convergenceRate(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    // Initialize benchmark variables
    Dictionary train_errors, test_errors;

    DataSet training_set, testing_set;
    splitDataSet(data_set, benchmark_options, &training_set, &testing_set);

    // reset parameter vector to forget previous training
    resetParameters(training_set, training_options);

    // print header of the csv
    maybePrintHeader("name,config params,epochs,train TPR,"
                     "train FPR,train Error,test TPR,test FPR,test Error\n");

    // shuffleKeyValue(data_points, labels, num_points_total, num_features);
    size_t total_epochs = 0;
    size_t epoch_step = training_options.num_epochs;
    training_options.num_epochs = 0;
    for (total_epochs = 0;
        total_epochs <= benchmark_options.max_num_epochs;
        total_epochs += epoch_step) {

        // training_options.num_epochs = total_epochs;
        training_function(training_set, training_options);
        training_options.num_epochs = epoch_step;
        training_options.config_params["curr_num_epochs"] = total_epochs;

        // Get Training error for this number of epochs
        initErrorDict(&train_errors);
        computeErrorRate(training_set, &train_errors);
        scaleErrorDict(&train_errors, 100.0);

        // Get Testing error for this number of epochs
        initErrorDict(&test_errors);
        computeErrorRate(testing_set, &test_errors);
        scaleErrorDict(&test_errors, 100.0);

        // output results after training
        printf( "%s,%s,%lu,%f,%f,%f,%f,%f,%f\n",
                name,
                configParamsToStr(training_options.config_params).c_str(),
                total_epochs,
                train_errors["tpr"],
                train_errors["fpr"],
                train_errors["error"],
                test_errors["tpr"],
                test_errors["fpr"],
                test_errors["error"]);
    }

    // reset configuration parameters
    *training_options.step_size =
        training_options.config_params["initial_step_size"];
    training_options.num_epochs = epoch_step;
}

// finds the time required to reach a given accuracy goal
void convergenceTime(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    // Initialize benchmark variables
    Dictionary train_errors, test_errors;

    DataSet training_set, testing_set;
    splitDataSet(data_set, benchmark_options, &training_set, &testing_set);

    // Ignore the first run to discard initialization overhead
    training_function(training_set, training_options);
    // reset parameter vector to forget previous training
    resetParameters(training_set, training_options);

    // find total number of epochs required to reach the accuracy goal
    size_t total_epochs = 0;
    training_options.num_epochs = 1;
    do {
        training_function(training_set, training_options);
        training_options.config_params["curr_num_epochs"] = total_epochs;
        total_epochs++;
    } while(computeErrorRate(training_set) > benchmark_options.error_goal
            && total_epochs < benchmark_options.max_num_epochs);

    training_options.num_epochs = total_epochs;

    // measure time needed to execute the number of epochs found above
    Timer training_timer(name);
    double training_time = 0.0;

    for (size_t k = 0; k < benchmark_options.num_runs; k++) {

        // reset parameter vector to forget previous training
        resetParameters(training_set, training_options);

        size_t total_epochs = 0;

        // shuffleKeyValue(data_points, labels, num_points_total, num_features);
        training_timer.start();
        training_function(training_set, training_options);
        training_time += training_timer.stop();

        // Get Training error
        computeErrorRate(training_set, &train_errors);

        // Get Testing error
        computeErrorRate(testing_set, &test_errors);
    }

    training_time /= benchmark_options.num_runs;
    scaleErrorDict(&train_errors, 100.0 / benchmark_options.num_runs);
    scaleErrorDict(&test_errors, 100.0 / benchmark_options.num_runs);

    // Output results
    maybePrintHeader("name,config params,runs,epochs,train time,train TPR,"
                     "train FPR,train Error,test TPR,test FPR,test Error\n");

    printf( "%s,%s,%lu,%lu,%f,%f,%f,%f,%f,%f,%f\n",
            name,
            configParamsToStr(training_options.config_params).c_str(),
            benchmark_options.num_runs,
            total_epochs,
            training_time,
            train_errors["tpr"],
            train_errors["fpr"],
            train_errors["error"],
            test_errors["tpr"],
            test_errors["fpr"],
            test_errors["error"]);

    // reset configuration parameters
    *training_options.step_size =
        training_options.config_params["initial_step_size"];
}
/******************************************************************************/
/*                                                                            */
/* Linear Algebra                                                             */
/*                                                                            */
/******************************************************************************/

#ifdef USE_OPENBLAS // ************************ using OpenBLAS *****************

#include "cblas.h"

// adds two vectors and stores the results in the first one
void addVectors(
    float* a,
    float* b,
    const size_t size,
    const float scale_for_b) {

    // for (size_t j = 0; j < size; j++) {
    //     a[j] += scale_for_b * b[j];
    // }

    cblas_saxpy(size, scale_for_b, b, 1, a, 1);
}

// computes dot product for two given vectors a and b
float dotProduct(
    float* a,
    float* b,
    const size_t size) {

    // float result = 0;
    // for (size_t j = 0; j < size; j++) {
    //     result += a[j] * b[j];
    // }
    // return result;

    return cblas_sdot(size, a, 1, b, 1);
}

// computes norm 2 of a given vector v
float norm2(float* v, const size_t size) {
    return sqrt(dotProduct(v, v, size));
}

// matrix-vector multiplication wrapper
void matrixVectorMultiply(
    float* matrix,
    float* vect,
    float scalar,
    size_t num_data_points,
    size_t num_features,
    float* result) {

    // for (size_t i = 0; i < num_data_points; i++) {
    //     addVectors(result, &matrix[i * num_features],
    //                  num_features, scalar * vect[i]);
    // }

    cblas_sgemv(CblasRowMajor, CblasTrans, num_data_points, num_features,
                scalar, matrix, num_features, vect, 1, 0, result, 1);
}

// updates the parameters (parameter_vector)
void updateParameters(
    FeatureType* parameter_vector,
    FeatureType* gradient,
    size_t num_features,
    double step_size,
    bool revert) {

    double sign = revert ? 1 : -1;
    step_size *= sign;

    // addVectors(parameter_vector, gradient, num_features, step_size);
    cblas_saxpy (num_features, step_size, gradient, 1, parameter_vector, 1);
}

#endif // ************************ end of OpenBLAS conditional *****************
