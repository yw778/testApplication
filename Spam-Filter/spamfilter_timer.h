#ifndef SPAMFILTER_TIMER
#define SPAMFILTER_TIMER

#include <time.h>

class Timer {
private:
    const char* name;
    clock_t start_time;
    double total_time;

public:
    Timer(const char* given_name = "--Undefined Name--");

    ~Timer();

    void start();

    void pause();

    void resume();

    double stop();
};



#endif
