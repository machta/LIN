#include "common.h"

#include <algorithm>

#include <cublas_v2.h>

#define A(r, c, h) A[(h)*(c) + (r)]
//#define A(r, c) A[n*(c) + (r)]

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
	
	cudaError_t error;
	float* Ad;
	int* infod;
	int* pivotd;
	error = cudaMalloc((void**) &Ad, n*n*sizeof(real));
	error = cudaMalloc((void**) &infod, sizeof(int));
	error = cudaMalloc((void**) &pivotd, n*sizeof(int));
	
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	
	printMatrix(n, n, A);
	
	auto start = high_resolution_clock::now();
	error = cudaMemcpy(Ad, A, n*n*sizeof(real), cudaMemcpyHostToDevice);
	
	//for (int i = 0; i < n*n; i++) A[i] = 0;
	
	printMatrix(n, n, A);
	
	stat = cublasSgetrfBatched(handle, n, &Ad, n, pivotd, infod, 1);
	
	cudaMemcpy(A, Ad, n*n*sizeof(real), cudaMemcpyDeviceToHost);
	int info = -1000000;
	printf("info = %d\n", info);
	cudaMemcpy(&info, infod, sizeof(int), cudaMemcpyDeviceToHost);
	printf("info = %d\n", info);
	int* pivot = new int[n];
	cudaMemcpy(pivot, pivotd, n*sizeof(int), cudaMemcpyDeviceToHost);
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, 1024);
	
	cublasDestroy(handle);
	cudaFree(Ad);
	cudaFree(infod);
	cudaFree(pivotd);
	delete[] pivot;
	delete[] A;
	delete[] x;
	return 0;
}

