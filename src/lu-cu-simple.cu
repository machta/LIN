#include "common.h"

#include <algorithm>

#include <cuda_runtime.h>

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

void syncDevice(float* hostP, float* deviceP, int m, int n)
{
	HANDLE_ERROR(cudaMemcpy(deviceP, hostP, m*n*sizeof(float), cudaMemcpyHostToDevice));
}

void syncHost(float* hostP, float* deviceP, int m, int n)
{
	HANDLE_ERROR(cudaMemcpy(hostP, deviceP, m*n*sizeof(float), cudaMemcpyDeviceToHost));
}

void printDeviceMatrix(int m, int n, float* A, float* Ad)
{
	syncHost(A, Ad, m, n);
	printMatrix(m, n, A);
}

__device__ void printSubMatrix(float* A, int m, int n, int h)
{
	int id = blockDim.x*threadIdx.y + threadIdx.x;
	
	if (id == 0)
	{	
		for (int j = 0; j < m; j++)
		{
			for (int i = 0; i < n; i++)
				printf("%7.2f", A[i*h + j]);
			printf("\n");
		}
		printf("\n");
	}
}

__device__ void copySubMatrix(float* src, float* dest, int m, int n, int srcH, int destH)
{
	if (threadIdx.x < m && threadIdx.y < n)
		dest[threadIdx.y*destH + threadIdx.x] = src[threadIdx.y*srcH + threadIdx.x];
}

__global__ void factorizeKernel(int m, int n, int h, float* A)
{
	extern __shared__ float C[];
	
	float* B = A + (blockIdx.x + 1)*n;
	int Cm = min(n, m - (blockIdx.x + 1)*n);
	m = n + Cm;
	
	copySubMatrix(A, C, n, n, h, m);
	copySubMatrix(B, C + n, Cm, n, h, m);
	
	__syncthreads();
	
	int id = blockDim.x*threadIdx.y + threadIdx.x;
	
	for (int i = 0; i < min(m - 1, n); ++i)
	{
    	float tmp = 1/C[i*m + i];
    	
    	int l = i + 1 + id;
    	if (l < m)
    		C[i*m + l] *= tmp;
    		
    	__syncthreads();
        
        /*#pragma omp for
		for (int l = i + 1; l < m; ++l)
			A(l, i, h) *= tmp;*/

		for (int k = i + 1 + threadIdx.y; k < n; k += blockDim.y)
			for (int l = i + 1 + threadIdx.x; l < m; l += blockDim.x)
				C[k*m + l] -= C[i*m + l]*C[k*m + i];

		__syncthreads();

		/*#pragma omp for
		for (int k = i + 1; k < n; ++k)
			for (int l = i + 1; l < m; ++l)
				A(l, k, h) -= A(l, i, h)*A(i, k, h);*/
	}
    
    //if (blockIdx.x == 0) printSubMatrix(C, m, n, m); __syncthreads();
    
	if (blockIdx.x == 0)
		copySubMatrix(C, A, n, n, m, h);
	copySubMatrix(C + n, B, Cm, n, m, h);
}

void factorize(int m, int n, int h, float* Ad)
{
	dim3 grid(max((m - 1)/n, 1));
	dim3 block(n, n);
	int size = 2*n*n*sizeof(float);
	
	factorizeKernel<<<grid, block, size>>>(m, n, h, Ad);
}

__global__ void updateRightKernel(int m, int n, int h, float* A)
{
	extern __shared__ float C[];
	
	float* B = A + (blockIdx.x + 1)*m*h;
	int Dn = min(m, n - (blockIdx.x + 1)*m);
	float* D = C + m*m;
	
	copySubMatrix(A, C, m, m, h, m);
	copySubMatrix(B, D, m, Dn, h, m);
	
	__syncthreads();
	
	if (threadIdx.y < Dn)
	{
		for (int k = 0; k < m - 1; ++k)
		{
			//if (blockIdx.x == 0) printSubMatrix(D, m, Dn, m); __syncthreads();
			
			int l = k + 1 + threadIdx.x;
			if (l < m)
				D[threadIdx.y*m + l] -= D[threadIdx.y*m + k]*C[k*m + l];
			
			__syncthreads();
		}
	}
	
	/*#pragma omp for
	for (int j = m; j < n; ++j)
	{
		for (int k = 0; k < m - 1; ++k)
			for (int l = k + 1; l < m; ++l)
				A(l, j, h) -= A(k, j, h)*A(l, k, h);
	}*/
    
    //if (blockIdx.x == 0) printSubMatrix(D, m, Dn, m); __syncthreads();
    
	copySubMatrix(D, B, m, Dn, m, h);
}

void updateRight(int m, int n, int h, float* Ad)
{
	dim3 grid((n - 1)/m);
	dim3 block(m, m);
	int size = 2*m*m*sizeof(float);
	
	updateRightKernel<<<grid, block, size>>>(m, n, h, Ad);
}

__global__ void updateDownKernel(int n, const int k, int h, float* A)
{
	extern __shared__ float B[];
	
	float* C = B + k*k;
	float* D = B + 2*k*k;
	int Cm = min(k, n - (blockIdx.x + 1)*k);
	int Dn = min(k, n - (blockIdx.y + 1)*k);
	
	float* Bglobal = A + (blockIdx.x + 1)*k + (blockIdx.y + 1)*k*h;
	float* Cglobal = A + (blockIdx.x + 1)*k;
	float* Dglobal = A + (blockIdx.y + 1)*k*h;
	
	copySubMatrix(Bglobal, B, Cm, Dn, h, k);
	copySubMatrix(Cglobal, C, Cm, k, h, k);
	copySubMatrix(Dglobal, D, k, Dn, h, k);
	
	__syncthreads();
	
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	if (x < Cm && y < Dn)
	{
		D += y*k;
		C += x;
		float tmp = 0;
		
		for (int i = 0; i < k; i++)
		{
			tmp += *C**D;
			C += k;
			D++;
		}
		
		B[y*k + x] -= tmp;
	}
	
	__syncthreads();
    
	copySubMatrix(B, Bglobal, Cm, Dn, k, h);
}

void updateDown(int n, int k, int h, float* Ad)
{
	dim3 grid((n - 1)/k, (n - 1)/k);
	dim3 block(k, k);
	int size = 3*k*k*sizeof(float);
	
	updateDownKernel<<<grid, block, size>>>(n, k, h, Ad);
}

void LU(int n, int k, float* A)
{
	float* Ad;
	HANDLE_ERROR(cudaMalloc(&Ad, n*n*sizeof(float)));
	
	syncDevice(A, Ad, n, n);
	
	for (int i = 0; i < n; i += k)
	{
		int w = std::min(k, n - i);
		float* leftTop = Ad + i*n + i;
		
		factorize(n - i, w, n, leftTop);
		
		//if (i < N - 1)
		{
			updateRight(k, n - i, n, leftTop);
			updateDown(n - i, k, n, leftTop);
		}
		
		//printDeviceMatrix(n, n, A, Ad);
	}
	
	syncHost(A, Ad, n, n);
	
	HANDLE_ERROR(cudaFree(Ad));
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
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	fprintf(stderr, "# Device name: %s\n", prop.name);

    using namespace std::chrono;

    real* A;
    int n;
    real* b;    
    int k = init(argc, argv, &n, &A, &b, false);
	real* x = new real[n];
	
	k = min(k, 32); // 32*32 is the maximum number of threads per block
	
	auto start = high_resolution_clock::now();
	LU(n, k, A);
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	//printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, k*k);
	
	delete[] A;
	delete[] x;
	return 0;
}

