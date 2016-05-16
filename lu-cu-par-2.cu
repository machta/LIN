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
	/*if (threadIdx.x < m && threadIdx.y < n)
		dest[threadIdx.y*destH + threadIdx.x] = src[threadIdx.y*srcH + threadIdx.x];*/
		
	/*for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			int x = 2*threadIdx.x + j;
			int y = 2*threadIdx.y + i;
			if (x < m && y < n)
				dest[y*destH + x] = src[y*srcH + x];
		}*/
		
	/*for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			int x = threadIdx.x + blockDim.x*j;
			int y = threadIdx.y + blockDim.y*i;
			if (x < m && y < n)
				dest[y*destH + x] = src[y*srcH + x];
		}*/
	
	for (int i = threadIdx.y; i < n; i += blockDim.y)
		for (int j = threadIdx.x; j < m; j += blockDim.x)
		{
			dest[i*destH + j] = src[i*srcH + j];
			
			//printf("%d %d\n", i*destH + j, i*srcH + j);
			
			//if (i*srcH + j > m*n)
				//printf("(%d,%d) src segfault", threadIdx.x, threadIdx.y);
			//if (i*destH + j > m*n)
				//printf("(%d,%d) dest segfault", threadIdx.x, threadIdx.y);
		}
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

const int UNROLL = 8;
const int K = 32;

__global__ void updateDownKernel(int n, int h, float* A)
{
	extern __shared__ float sharedMem[];
	
	float* C = sharedMem;
	float* D = sharedMem + K*K;
	int Cm = min(K, n - (blockIdx.x + 1)*K);
	int Dn = min(2*K, n - (blockIdx.y*2 + 1)*K);
	
	float* Bglobal = A + (blockIdx.x + 1)*K + (blockIdx.y*2 + 1)*K*h;
	float* Cglobal = A + (blockIdx.x + 1)*K;
	float* Dglobal = A + (blockIdx.y*2 + 1)*K*h;
	
	copySubMatrix(Cglobal, C, Cm, K, h, K);
	copySubMatrix(Dglobal, D, K, Dn, h, K);
	
	__syncthreads();
	
	/*if (blockIdx.x == 0 && blockIdx.y == 0) printSubMatrix(Bglobal, Cm, Dn, h); __syncthreads();
	if (blockIdx.x == 0 && blockIdx.y == 0) printSubMatrix(C, Cm, K, K); __syncthreads();
	if (blockIdx.x == 0 && blockIdx.y == 0) printSubMatrix(D, K, Dn, K); __syncthreads();*/
	
	float tmp[UNROLL];
	for (int i = 0; i < UNROLL; i++)
		tmp[i] = 0;
	
	int threads = blockDim.x*blockDim.y;
	
	D += threadIdx.y*K;
	C += threadIdx.x;
	
	for (int j = 0; j < K; j++)
	{
		for (int i = 0; i < UNROLL; i++) 
			tmp[i] += C[0]*D[i*threads];
		C += K;
		D++;
	}
	
	for (int i = 0; i < UNROLL; i++)
	{
		//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
			//printf("%f\n", tmp[i]); __syncthreads();
		
		int x = threadIdx.x;
		int y = threadIdx.y + i*blockDim.y;
		if (x < Cm && y < Dn)
		{
			//if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
				//printf("%f\n", Bglobal[y*h + x]); __syncthreads();
		
			Bglobal[y*h + x] -= tmp[i];
			
			//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
				//printf("%f\n", Bglobal[y*h + x]); __syncthreads();
		}
	}
}

void updateDown(int n, int k, int h, float* Ad)
{
	dim3 grid((n - 1)/k, ((n - 1)/k + 1)/2);
	//dim3 grid(1,2);
	dim3 block(32, 8);
	
	updateDownKernel<<<grid, block, 3*32*8*8*sizeof(float)>>>(n, h, Ad);
}

void LU(int n, int k, float* A)
{
	float* Ad;
	HANDLE_ERROR(cudaMalloc(&Ad, n*n*sizeof(float)));
	
	syncDevice(A, Ad, n, n);
	
	for (int i = 0; i < n; i += k)
	{
		//printf("i=%d\n", i);
		
		int w = std::min(k, n - i);
		float* leftTop = Ad + i*n + i;
		
		factorize(n - i, w, n, leftTop);
		
		//if (i < N - 1)
		{
			updateRight(k, n - i, n, leftTop);
			//printDeviceMatrix(n, n, A, Ad);
			updateDown(n - i, k, n, leftTop);
		}
		
		//printDeviceMatrix(n, n, A, Ad);
		//if (i > 0) break;
	}
	
	//printf("before the end\n");
	
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
    using namespace std::chrono;

    real* A;
    int n;
    real* b;
    int k = init(argc, argv, &n, &A, &b, false);
	real* x = new real[n];
	
	//k = min(k, 32);
	k = 32;
	
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

