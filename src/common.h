#ifndef COMMON_H
#define COMMON_H

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <cassert>
#include <string>
#include <chrono>

#if defined USE_DOUBLE
#define real double
#else
#define real float
#endif

inline void error(std::string msg)
{
	fprintf(stderr, "ERROR: %s\n", msg.c_str());
	exit(1);
}

void printResult(int n, real* x, long long ns, double flopCount, int k, int maxThreads);

int init(int argc, char** argv, int* n, real** A, real** b, bool tiled);

void column2Tiled(int *r, int *c, int n, int k);

void printMatrix(int n, int k, real* A);

#endif // COMMON_H

