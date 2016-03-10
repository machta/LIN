#ifndef COMMON_H
#define COMMON_H

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <cassert>
#include <string>
#include <chrono>

using namespace std;

inline void error(string msg)
{
	fprintf(stderr, "ERROR: %s\n", msg.c_str());
	exit(1);
}

void printResult(int n, real* x, long long ns, real flopCount);

int init(int argc, char** argv, int* n, real** A, real** b);

#endif // COMMON_H

