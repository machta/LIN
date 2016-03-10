#include "common.h"

#define A(r, c) A[N*(c) + (r)]

namespace
{

int alloc(int n, real** A, real** b)
{
	const int colAlign = 4;	(void)colAlign;
	
	int N = n;
	//N = (N + colAlign - 1)/colAlign*colAlign;
	*A = new real[N*n + N];
	*b = *A + N*n;
	return N;
}

// Load the values from stdin.
void initFromInput(int n, int N, real* A, real* b)
{
	float tmp;

	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i)
		{
			int res = scanf("%f", &tmp);
			A[N*j + i] = tmp;
			(void)res;
		}

	for (int i = 0; i < n; ++i)
	{
		int res = scanf("%f", &tmp);
		b[i] = tmp;
		(void)res;
	}
}

// Create a random symmetric positive definit matrix A and vector b.
void initRandom(int n, int N, real* A, real* b)
{
	srand (time(NULL));
	const real scale = 100./RAND_MAX;
	
	real* columnSums = new real[n];
	
	for (int j = 0; j < n; ++j)
		columnSums[j] = 0;
	
	for (int j = 0; j < n; ++j)
	{
		for (int i = j + 1; i < n; ++i)
		{
			columnSums[j] += columnSums[i] += A[N*j + i] = A[N*i + j] = rand()*scale;
		}
	
		A[N*j + j] = 2*columnSums[j] + 1;
		b[j] = rand()*scale;
	}
	
	delete[] columnSums;
}

} // namespace

int init(int argc, char** argv, int* n, real** A, real** b)
{
	if (argc <= 1)
		error("not enough parameters.");

	int N;	
	
	if (argv[1][0] == 'i')
	{
		int res = scanf("%d", n);
		(void)res;
		N = alloc(*n, A, b);
		initFromInput(*n, N, *A, *b);
	}
	else if (argv[1][0] == 'r')
	{
		if (argc <= 2)
			error("not enough parameters.");
			
		*n = stoi(argv[2]);
		N = alloc(*n, A, b);
		initRandom(*n, N, *A, *b);
	}
	else
	{
		error("only 'r' and 'i' modes are allowed.");
	}
	
	return N;
}

void printResult(int n, real* x, long long ns, real flopCount)
{
	for (int i = 0; i < n; ++i)
	    printf("%f\n", x[i]);
	    
	real giga = 1000*1000*1000;
	real seconds = ns/giga;
	fprintf(stderr, "Seconds elapsed: %f\nGFLOPS: %f\n", seconds, flopCount/giga/seconds);
}

