#include "common.h"

#include <algorithm>

#if defined USE_BLAS
#include <cblas.h>
#endif

#define A(r, c, h) A[(h)*(c) + (r)]
//#define A(r, c) A[n*(c) + (r)]

void factorize(int m, int n, int h, real* A)
{
    for (int i = 0; i < std::min(m - 1, n); ++i)
    {
    	real tmp = 1/A(i, i, h);
        
        #pragma omp for
		for (int l = i + 1; l < m; ++l)
			A(l, i, h) *= tmp;

		#pragma omp for
		for (int k = i + 1; k < n; ++k)
		{
			real tmp2 = A(i, k, h);
			for (int l = i + 1; l < m; ++l)
				A(l, k, h) -= A(l, i, h)*tmp2;
		}
    }
}

void updateRight(int m, int n, int h, real* A)
{
	#pragma omp for
	for (int j = m; j < n; ++j)
	{
		for (int k = 0; k < m - 1; ++k)
		{
			real tmp = A(k, j, h);
			for (int l = k + 1; l < m; ++l)
				A(l, j, h) -= tmp*A(l, k, h);
		}
	}
}

// B = B - C*D;
void MMMS(int m, int n, int o, int h, real* __restrict__ B, real* __restrict__ C, real* __restrict__ D)
{
#if defined USE_BLAS
#if defined USE_DOUBLE
	cblas_dgemm
#else
	cblas_sgemm
#endif
	(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, o, -1, C, h, D, h, 1, B, h);
#else
	// Explicit vectorization.
	for (int i = 0; i < n; ++i) // col
		for (int k = 0; k < o; ++k) // product
		{
			real tmp = D[h*i + k];
			for (int j = 0; j < m; ++j) // row
				B[h*i + j] -= C[h*k + j]*tmp;
		}
#endif
}

void updateDown(int n, int k, int h, real* A)
{
	#pragma omp for collapse(2)
	for (int i = k; i < n; i += k)
		for (int j = k; j < n; j += k)
		{
			int w = std::min(k, n - i);
			
			real* B = &A(j, i, h);
			real* C = &A(j, 0, h);
			real* D = &A(0, i, h);
			
			MMMS(std::min(k, n - j), w, k, h, B, C, D);
		}
}

void LU(int n, int k, real* A)
{
	#pragma omp parallel
	for (int i = 0; i < n; i += k)
	{
		int w = std::min(k, n - i);
		real* leftTop = &A(i, i, n);
		
		factorize(n - i, w, n, leftTop);
		
		//if (i < N - 1)
		{
			updateRight(k, n - i, n, leftTop);
			updateDown(n - i, k, n, leftTop);
		}
		
		//printMatrix(n, n, A);	fprintf(stderr, "\n");
	}
}

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
    int k = init(argc, argv, &n, &A, &b, false);
	real* x = new real[n];
	
	auto start = high_resolution_clock::now();
	LU(n, k, A);
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	//printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k);
	
	delete[] A;
	delete[] x;
	return 0;
}

