#ifndef SPAMFILTER_DEFS
#define SPAMFILTER_DEFS

#define NUM_FEATURES      1024
#define NUM_SAMPLES       5000
#define NUM_TRAINING      4500
#define NUM_TESTING       (NUM_SAMPLES - NUM_TRAINING)
#define NUM_VALIDATION    0
#define X_SIZE            (NUM_FEATURES * NUM_SAMPLES)
#define Y_SIZE            (NUM_SAMPLES)
#define LAMBDA            0 //regularization parameter
#define STEP_SIZE         60000 //step size (eta)
#define EPSILON           0.0000001
#define MAX_ITER          (NUM_TRAINING)
#define TOLERANCE         0.0001
#define NUM_EPOCHS        5
#define NUM_RUNS          3

typedef float FeatureType;
typedef float LabelType;

#define USE_COLORED_OUTPUT //comment out this line to remove color from output
#ifdef USE_COLORED_OUTPUT
// text colors
    #define ANSI_COLOR_RED     "\x1b[31m"
    #define ANSI_COLOR_GREEN   "\x1b[32m"
    #define ANSI_COLOR_YELLOW  "\x1b[33m"
    #define ANSI_COLOR_BLUE    "\x1b[34m"
    #define ANSI_COLOR_MAGENTA "\x1b[35m"
    #define ANSI_COLOR_CYAN    "\x1b[36m"
    #define ANSI_COLOR_RESET   "\x1b[0m"
#else
    #define ANSI_COLOR_RED     ""
    #define ANSI_COLOR_GREEN   ""
    #define ANSI_COLOR_YELLOW  ""
    #define ANSI_COLOR_BLUE    ""
    #define ANSI_COLOR_MAGENTA ""
    #define ANSI_COLOR_CYAN    ""
    #define ANSI_COLOR_RESET   ""
#endif

#endif
