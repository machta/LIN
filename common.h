#ifndef COMMON_H
#define COMMON_H

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <cassert>
#include <string>
#include <chrono>

using namespace std;

const int colAlign = 4;

inline void error(string msg)
{
	fprintf(stderr, "ERROR: %s\n", msg.c_str());
	exit(1);
}

void printResult(int n, double* x, long long ns, double flopCount);

int init(int argc, char** argv, int* n, double** A, double** b);

#endif // COMMON_H

