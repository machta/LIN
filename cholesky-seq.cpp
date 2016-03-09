#include "common.h"

#include <cmath>

#define A(r, c) A[N*(c) + (r)]

void cholesky(int n, int N, double* A)
{
    for (int j = 0; j < n; ++j)
    {
    	double tmp = A(j, j);
    	
    	for (int k = 0; k < j; ++k)
    		tmp -= A(j, k)*A(j, k);
    		
    	A(j, j) = sqrt(tmp);
    	
    	for (int i = j + 1; i < n; ++i)
    	{
    		tmp = A(i, j);
    	
			for (int k = 0; k < j; ++k)
				tmp -= A(i, k)*A(j, k);
				
			A(i, j) = tmp/A(j, j);
    	}
    }
}

// Solve Lx = b for x.
void forwardSubstitution(int n, int N, double* A, double* x, double* b)
{
	for (int i = 0; i < n; ++i)
	{
		float sum = b[i];
		for (int j = 0; j < i; ++j)
			sum -= A(i, j)*x[j];
		x[i] = sum/A(i, i);
	}
}

// Solve L'x = b for x.
void backwardSubstitution(int n, int N, double* A, double* x, double* b)
{
	for (int i = n - 1; i >= 0; --i)
	{
		float sum = b[i];
		for (int j = i + 1; j < n; ++j)
			sum -= A(j, i)*x[j];
		x[i] = sum/A(i, i);
	}
}

int main(int argc, char** argv)
{
    using namespace std::chrono;

    double* A;
    int n;
    double* b;
    int N = init(argc, argv, &n, &A, &b);
	double* x = new double[n];
	
	auto start = high_resolution_clock::now();
	
	cholesky(n, N, A);
	
	forwardSubstitution(n, N, A, x, b);
	backwardSubstitution(n, N, A, b, x);
	
	auto end = high_resolution_clock::now();	
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 1./3*n*n*n);
	
	delete[] A;
	delete[] x;
	return 0;
}

