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
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	if (x < m && y < n)
		dest[y*destH + x] = src[y*srcH + x];
}

template <int BLOCK_SIZE>
__global__ void factorizeKernel(int m, int n, int h, float* A)
{
	__shared__ float C[2*BLOCK_SIZE*BLOCK_SIZE];
	
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
        
		for (int k = i + 1 + threadIdx.y; k < n; k += blockDim.y)
			for (int l = i + 1 + threadIdx.x; l < m; l += blockDim.x)
				C[k*m + l] -= C[i*m + l]*C[k*m + i];

		__syncthreads();
	}
    
    //if (blockIdx.x == 0) printSubMatrix(C, m, n, m); __syncthreads();
    
	if (blockIdx.x == 0)
		copySubMatrix(C, A, n, n, m, h);
	copySubMatrix(C + n, B, Cm, n, m, h);
}

template <int BLOCK_SIZE>
void factorize(int m, int n, int h, float* Ad)
{
	dim3 grid(max((m - 1)/n, 1));
	dim3 block(n, n);
	
	factorizeKernel<BLOCK_SIZE><<<grid, block>>>(m, n, h, Ad);
}

template <int BLOCK_SIZE>
__global__ void updateRightKernel(int n, int h, float* A)
{
	__shared__ float C[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float D[BLOCK_SIZE*BLOCK_SIZE];
	
	float* B = A + (blockIdx.x + 1)*BLOCK_SIZE*h;
	int Dn = min(BLOCK_SIZE, n - (blockIdx.x + 1)*BLOCK_SIZE);
	
	copySubMatrix(A, C, BLOCK_SIZE, BLOCK_SIZE, h, BLOCK_SIZE);
	copySubMatrix(B, D, BLOCK_SIZE, Dn, h, BLOCK_SIZE);
	
	__syncthreads();
	
	if (threadIdx.y < Dn)
	{
		for (int k = 0; k < BLOCK_SIZE - 1; ++k)
		{
			//if (blockIdx.x == 0) printSubMatrix(D, BLOCK_SIZE, Dn, BLOCK_SIZE); __syncthreads();
			
			int l = k + 1 + threadIdx.x;
			if (l < BLOCK_SIZE)
				D[threadIdx.y*BLOCK_SIZE + l] -= D[threadIdx.y*BLOCK_SIZE + k]*C[k*BLOCK_SIZE + l];
			
			__syncthreads();
		}
	}
	//if (blockIdx.x == 0) printSubMatrix(D, BLOCK_SIZE, Dn, BLOCK_SIZE); __syncthreads();
    
	copySubMatrix(D, B, BLOCK_SIZE, Dn, BLOCK_SIZE, h);
}

template <int BLOCK_SIZE>
void updateRight(int n, int h, float* Ad)
{
	dim3 grid((n - 1)/BLOCK_SIZE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	updateRightKernel<BLOCK_SIZE><<<grid, block>>>(n, h, Ad);
}

template <int BLOCK_SIZE, int UNROLL>
__device__ void copySubMatrix(float* src, float* dest, int m, int n, int srcH)
{
	int x = threadIdx.x;
	for (int i = 0; i < UNROLL; i++)
	{
		int y = threadIdx.y + i*BLOCK_SIZE/UNROLL;
	
		if (x < m && y < n)
			dest[y*BLOCK_SIZE + x] = src[y*srcH + x];
	}
}

template <int BLOCK_SIZE, int UNROLL>
__global__ void updateDownKernel(int n, int h, float* A)
{
	__shared__ float C[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float D[BLOCK_SIZE*BLOCK_SIZE];
	
	int Cm = min(BLOCK_SIZE, n - (blockIdx.x + 1)*BLOCK_SIZE);
	int Dn = min(BLOCK_SIZE, n - (blockIdx.y + 1)*BLOCK_SIZE);
	
	copySubMatrix<BLOCK_SIZE, UNROLL>(A + (blockIdx.x + 1)*BLOCK_SIZE, C, Cm, BLOCK_SIZE, h);
	copySubMatrix<BLOCK_SIZE, UNROLL>(A + (blockIdx.y + 1)*BLOCK_SIZE*h, D, BLOCK_SIZE, Dn, h);
	
	A += (blockIdx.x + 1)*BLOCK_SIZE + (blockIdx.y + 1)*BLOCK_SIZE*h;
	
	__syncthreads();
	
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	/*if (blockIdx.x == 1 && blockIdx.y == 0 && x == 0 && y == 0)
	{
		printSubMatrix(A, Cm, Dn, h);
		printSubMatrix(C, Cm, BLOCK_SIZE, BLOCK_SIZE);
		printSubMatrix(D, BLOCK_SIZE, Dn, BLOCK_SIZE);
	}
	__syncthreads();*/
	
	float tmp[UNROLL];
	for (int i = 0; i < UNROLL; i++)
		tmp[i] = 0;
	
	for (int j = 0; j < BLOCK_SIZE; j++)
	{
		for (int i = 0; i < UNROLL; i++)
			tmp[i] += C[j*BLOCK_SIZE + x]*D[(y + i*BLOCK_SIZE/UNROLL)*BLOCK_SIZE + j];
	}
	
	for (int i = 0; i < UNROLL; i++)
	{
		y = threadIdx.y + i*BLOCK_SIZE/UNROLL;
		
		if (x < Cm && y < Dn)
		{
			//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && threadIdx.y == 1)
				//printf("%.02f - %.02f\n", A[y*h + x], tmp[i]);
				
			A[y*h + x] -= tmp[i];
		}
	}
}

#define CASE(i) case i: updateDownKernel<BLOCK_SIZE, i><<<grid, block>>>(n, h, Ad); break;

template <int BLOCK_SIZE>
void updateDown(int n, int h, float* Ad, int t)
{
	dim3 grid((n - 1)/BLOCK_SIZE, (n - 1)/BLOCK_SIZE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE/t);
	
	switch (t)
	{
		CASE(1);
		CASE(2);
		CASE(4);
		CASE(8);
		CASE(16);
		CASE(32);
	}
}

#undef CASE

template <int BLOCK_SIZE>
void LU(int n, int t, float* A)
{
	float* Ad;
	HANDLE_ERROR(cudaMalloc(&Ad, n*n*sizeof(float)));
	
	syncDevice(A, Ad, n, n);
	
	for (int i = 0; i < n; i += BLOCK_SIZE)
	{
		int w = std::min(BLOCK_SIZE, n - i);
		float* leftTop = Ad + i*n + i;
		
		factorize<BLOCK_SIZE>(n - i, w, n, leftTop);
		
		//if (i < N - 1)
		{
			updateRight<BLOCK_SIZE>(n - i, n, leftTop);
			//printDeviceMatrix(n, n, A, Ad);
			updateDown<BLOCK_SIZE>(n - i, n, leftTop, t);
		}
		
		//printDeviceMatrix(n, n, A, Ad);
		//if (i > 0) break;
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

// Make sure it's <= 32 and a power of 2.
int adjustK(int v)
{
	v = min(v, 32);
	v--;
	v |= v >> 1;
	v |= v >> 2;
//	v |= v >> 4;
//	v |= v >> 8;
//	v |= v >> 16;
	v++;
	return v;
}

int main(int argc, char** argv)
{
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# Device name: %s\n", prop.name);
    
    using namespace std::chrono;

    real* A;
    int n;
    real* b;    
    int k = init(argc, argv, &n, &A, &b, false);
	real* x = new real[n];
	
	k = adjustK(k);
	
	auto start = high_resolution_clock::now();
	LU<32>(n, k, A);
	auto end = high_resolution_clock::now();
	
	forwardSubstitution(n, A, x, b);
	backwardSubstitution(n, A, b, x);
	
	//printMatrix(n, n, A);
	
	nanoseconds elapsedTime = end - start;
	printResult(n, b, elapsedTime.count(), 2./3*n*n*n, k, 32*32/k);
	
	delete[] A;
	delete[] x;
	return 0;
}

