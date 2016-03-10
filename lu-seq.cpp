#include "common.h"

#define A(r, c) A[N*(c) + (r)]

void LU(int n, int N, real* A)
{
    for (int i = 0; i < n - 1; ++i)
    {
        real tmp = 1/A(i, i);
        
        for (int j = i + 1; j < n; ++j)
            A(j, i) *= tmp;
        
        /*for (int j = i + 1; j < n; ++j)
            for (int k = i + 1; k < n; ++k)
                A(j, k) -= A(j, i)*A(i, k);*/
            
        for (int k = i + 1; k < n; ++k)
            for (int j = i + 1; j < n; ++j)
                A(j, k) -= A(j, i)*A(i, k);
    }
}

// Solve Lx = b for x.
void forwardSubstitution(int n, int N, real* A, real* x, real* b)
{
	for (int i = 0; i < n; ++i)
	{
		real sum = b[i];
		for (int j = 0; j < i; ++j)
			sum -= A(i, j)*x[j];
		x[i] = sum;
	}
}

// Solve Ux = b for x.
void backwardSubstitution(int n, int N, real* A, real* x, real* b)
{
	for (int i = n - 1; i >= 0; --i)
	{
		real sum = b[i];
		for (int j = i + 1; j < n; ++j)
			sum -= A(i, j)*x[j];
		x[i] = sum/A(i, i);
	}
}

int main(int argc, char** argv)
{
    using namespace std::chrono;

    real* A;
    int n;
    real* b;
    int N = init(argc, argv, &n, &A, &b);
	real* x = new real[n];
	
	auto start = high_resolution_clock::now();
	
	LU(n, N, A);
	
	forwardSubstitution(n, N, A, x, b);
	backwardSubstitution(n, N, A, b, x);
	
	auto end = high_resolution_clock::now();	
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n);
	
	delete[] A;
	delete[] x;
	return 0;
}

