#include "common.h"

#include <algorithm>

#include <omp.h>

#if defined USE_BLAS
#include <cblas.h>
#endif

#define A(r, c, h) A[(h)*(c) + (r)]

void factorize(int m, int n, real* AA)
{
	int M = (m + n - 1)/n;

    for (int i = 0; i < std::min(m - 1, n); ++i)
    {
    	real tmp = 1/AA[n*i + i];
        
        #pragma omp for
        for (int j = 0; j < M; ++j)
		{
			real* A = AA;
			if (j == 0)
			{
				for (int l = i + 1; l < n; ++l)
					A(l, i, n) *= tmp;
		
				for (int k = i + 1; k < n; ++k)
					for (int l = i + 1; l < n; ++l)
						A(l, k, n) -= A(l, i, n)*A(i, k, n);
			}
			else
			{
				A += n*n*j;
				int h = std::min(n, m - j*n);			
		
				for (int l = 0; l < h; ++l)
					A(l, i, h) *= tmp;
		
				for (int k = i + 1; k < n; ++k)
					for (int l = 0; l < h; ++l)
						A(l, k, h) -= A(l, i, h)*AA[n*k + i]/*A(i, k, h)*/;
			}
		}
    }
}

void updateRight(int m, int n, int h, real* A)
{
	int N = (n + m - 1)/m;
	
	#pragma omp for
	for (int i = 1; i < N; ++i)
	{
		int w = std::min(m, n - i*m);
		real* B = A + h*m*i - (m - w)*(h - n);
		
		for (int j = 0; j < w; ++j)
		{
			for (int k = 0; k < m - 1; ++k)
				for (int l = k + 1; l < m; ++l)
					B[m*j + l] -= B[m*j + k]*A(l, k, m);
		}
	}
}

// B = B - C*D;
void MMMS(int m, int n, int h, real* __restrict__ B, real* __restrict__ C, real* __restrict__ D)
{
#if defined USE_BLAS
#if defined USE_DOUBLE
	cblas_dgemm
#else
	cblas_sgemm
#endif
	(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, h, -1, C, m, D, h, 1, B, m);
#else
	/*for (int i = 0; i < n; ++i) // col
		for (int k = 0; k < h; ++k) // product
			for (int j = 0; j < m; ++j) // row
				B[m*i + j] -= C[m*k + j]*D[h*i + k];*/
	
	// Explicit vectorization.
	for (int i = 0; i < n; ++i) // col
		for (int k = 0; k < h; ++k) // product			
		{
			real tmp = D[h*i + k];
			for (int j = 0; j < m; ++j) // row
				B[m*i + j] -= C[m*k + j]*tmp;
		}
#endif
}

void updateDown(int n, int k, int h, real* A)
{
	int N = (n + k - 1)/k;
	
	#pragma omp for collapse(2)
	for (int i = 1; i < N; ++i)
		for (int j = 1; j < N; ++j)
		{
			int w = std::min(k, n - i*k);
			int iO = h*k*i - (k - w)*(h - n);
			
			real* B = A + iO + w*k*j;
			real* C = A + k*k*j;
			real* D = A + iO;
			
			MMMS(std::min(k, n - j*k), w, k, B, C, D);
		}
}

void LU(int n, int k, real* A)
{
	int N = (n + k - 1)/k;
	
	#pragma omp parallel
	for (int i = 0; i < N; ++i)
	{
		int w = std::min(k, n - i*k);
		real* leftTop = A + i*(n*k + w*k);
		
		factorize(n - i*k, w, leftTop);
		
		//if (i < N - 1)
		{
			updateRight(k, n - i*k, n, leftTop);
			updateDown(n - i*k, k, n, leftTop);
		}
		
		//printMatrix(n, k, A);	fprintf(stderr, "\n");
	}
}

/*// Solve Lx = b for x.
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
}*/

// Solve Lx = b for x.
void forwardSubstitution(int n, int k, real* A, real* x, real* b)
{
	for (int i = 0; i < n; ++i)
	{
		real sum = b[i];
		for (int j = 0; j < i; ++j)
		{
			int r = i, c = j;
        	column2Tiled(&r, &c, n, k);
			sum -= A(r, c, n)*x[j];
		}
		x[i] = sum;
	}
}

// Solve Ux = b for x.
void backwardSubstitution(int n, int k, real* A, real* x, real* b)
{
	for (int i = n - 1; i >= 0; --i)
	{
		real sum = b[i];
		for (int j = i + 1; j < n; ++j)
		{
			int r = i, c = j;
        	column2Tiled(&r, &c, n, k);
			sum -= A(r, c, n)*x[j];
		}
		
		int r = i, c = i;
        column2Tiled(&r, &c, n, k);
		x[i] = sum/A(r, c, n);
	}
}

int main(int argc, char** argv)
{
    using namespace std::chrono;

    real* A;
    int n;
    real* b;    
    int k = init(argc, argv, &n, &A, &b, true);
	real* x = new real[n];
	
	auto start = high_resolution_clock::now();
	LU(n, k, A);
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, k, A, x, b);
	backwardSubstitution(n, k, A, b, x);
	
	//printMatrix(n, k, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, omp_get_max_threads());
	
	delete[] A;
	delete[] x;
	return 0;
}

