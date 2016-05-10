#include "common.h"

#include <algorithm>

#include <cusolverDn.h>

#define A(r, c, h) A[(h)*(c) + (r)]
//#define A(r, c) A[n*(c) + (r)]

#define HANDLE_ERROR(err) HandleError(err, __FILE__, __LINE__)

void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

void LU(int n, real* A)
{
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	
	float* Ad;
	HANDLE_ERROR(cudaMalloc(&Ad, n*n*sizeof(float)));
	
	int Lwork;
	cusolverDnSgetrf_bufferSize(handle, n, n, Ad, n, &Lwork);	
	float* workspace;
	HANDLE_ERROR(cudaMalloc(&workspace, Lwork*sizeof(float)));
	
	int* pivot;
	HANDLE_ERROR(cudaMalloc(&pivot, n*sizeof(int)));
	
	int* info;
	HANDLE_ERROR(cudaMalloc(&info, sizeof(int)));
	
	HANDLE_ERROR(cudaMemcpy(Ad, A, n*n*sizeof(float), cudaMemcpyHostToDevice));
	cusolverDnSgetrf(handle, n, n, Ad, n, workspace, pivot, info);
	HANDLE_ERROR(cudaMemcpy(A, Ad, n*n*sizeof(float), cudaMemcpyDeviceToHost));
	
	cusolverDnDestroy(handle);
	HANDLE_ERROR(cudaFree(Ad));
	HANDLE_ERROR(cudaFree(workspace));
	HANDLE_ERROR(cudaFree(pivot));
	HANDLE_ERROR(cudaFree(info));
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
	LU(n, A);	
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	//printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, 1024);
	
	delete[] A;
	delete[] x;
	return 0;
}

