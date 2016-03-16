#include "common.h"

#include <algorithm>

#if defined USE_BLAS
#include <cblas.h>
#endif

#define A(r, c, h) A[(h)*(c) + (r)]

// Solve Lx = b for x.
void forwardSubstitution(int n, real* A, real* x, real* b)
{
	for (int i = 0; i < n; ++i)
	{
		real sum = b[i];
		for (int j = 0; j < i; ++j)
			sum -= A(i, j, n)*x[j];
		x[i] = sum;
	}
}

// Solve Ux = b for x.
void backwardSubstitution(int n, real* A, real* x, real* b)
{
	for (int i = n - 1; i >= 0; --i)
	{
		real sum = b[i];
		for (int j = i + 1; j < n; ++j)
			sum -= A(i, j, n)*x[j];
		x[i] = sum/A(i, i, n);
	}
}

int main(int argc, char** argv)
{
	using namespace std::chrono;

	real* A;
	int n;
	real* b;    
	int k = init(argc, argv, &n, &A, &b);
	real* x = new real[n];

	auto start = high_resolution_clock::now();

#if defined USE_BLAS
#if defined USE_DOUBLE
	cblas_dgetrf
#else
	cblas_sgetrf
#endif
	
#endif

	forwardSubstitution(n, k, A, x, b);
	backwardSubstitution(n, k, A, b, x);

	auto end = high_resolution_clock::now();	

	//printMatrix(n, k, A);

	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, n);

	delete[] A;
	delete[] x;
	return 0;
}

