#include "common.h"

#define A(r, c) A[N*(c) + (r)]

void gauss(int n, int N, double* A)
{
    for (int i = 0; i < n - 1; ++i)
    {
        double tmp = 1/A(i, i);
        
        for (int j = i + 1; j < n; ++j)
            A(j, i) *= tmp;
        
        // Here include the last column b otherwise the same as LU.      
        for (int k = i + 1; k <= n; ++k) 
            for (int j = i + 1; j < n; ++j)
                A(j, k) -= A(j, i)*A(i, k);
    }
}

// Solve Ux = b for x.
void backwardSubstitution(int n, int N, double* A, double* x, double* b)
{
	for (int i = n - 1; i >= 0; --i)
	{
		float sum = b[i];
		for (int j = i + 1; j < n; ++j)
			sum -= A(i, j)*x[j];
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
	
	gauss(n, N, A);
	
	backwardSubstitution(n, N, A, x, b);
	
	auto end = high_resolution_clock::now();	
	
	nanoseconds elapsedTime = end - start;
	printResult(n, x, elapsedTime.count(), 2./3*n*n*n);
	
	delete[] A;
	delete[] x;
	return 0;
}

