#include "common.h"

#include <algorithm>

#include <cublas_v2.h>

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
	
	float* Ad;
	float** array;
	int* infod;
	HANDLE_ERROR(cudaMalloc(&Ad, n*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&array, sizeof(float*)));
	HANDLE_ERROR(cudaMalloc(&infod, sizeof(int)));
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	//printMatrix(n, n, A);
	
	auto start = high_resolution_clock::now();
	
	HANDLE_ERROR(cudaMemcpy(Ad, A, n*n*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(array, &Ad, sizeof(float*), cudaMemcpyHostToDevice));

	cublasSgetrfBatched(handle, n, array, n, nullptr, infod, 1);
	
	HANDLE_ERROR(cudaMemcpy(A, Ad, n*n*sizeof(float), cudaMemcpyDeviceToHost));
	
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	//printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, 1024);
	
	cublasDestroy(handle);
	HANDLE_ERROR(cudaFree(Ad));
	HANDLE_ERROR(cudaFree(array));
	HANDLE_ERROR(cudaFree(infod));
	delete[] A;
	delete[] x;
	return 0;
}

