#include <stdio.h>
#include "spamfilter_timer.h"

// time in miliseconds

Timer::Timer(const char* given_name) : name(given_name) {
    total_time = 0;
}

Timer::~Timer() {
}

void Timer::start() {
    start_time = clock();
}

void Timer::pause() {
    total_time += (clock() - start_time) * 1000.0 / (double) CLOCKS_PER_SEC;
}

void Timer::resume() {
    start_time = clock();
}

double Timer::stop() {
    Timer::pause();
    double return_time = total_time;
    total_time = 0;
    return return_time;
}
