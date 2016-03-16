#include "common.h"

#define A(r, c) A[n*(c) + (r)]

void gauss(int n, real* A)
{
    for (int i = 0; i < n - 1; ++i)
    {
        real tmp = 1/A(i, i);
        
        for (int j = i + 1; j < n; ++j)
            A(j, i) *= tmp;
        
        // Here include the last column b otherwise the same as LU.      
        for (int k = i + 1; k <= n; ++k) 
            for (int j = i + 1; j < n; ++j)
                A(j, k) -= A(j, i)*A(i, k);
    }
}

// Solve Ux = b for x.
void backwardSubstitution(int n, real* A, real* x, real* b)
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
    init(argc, argv, &n, &A, &b);
	real* x = new real[n];
	
	auto start = high_resolution_clock::now();
	
	gauss(n, A);
	
	backwardSubstitution(n, A, x, b);
	
	auto end = high_resolution_clock::now();	
	
	nanoseconds elapsedTime = end - start;
	printResult(n, x, elapsedTime.count(), 2./3*n*n*n, n);
	
	delete[] A;
	delete[] x;
	return 0;
}

