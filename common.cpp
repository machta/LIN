#include "common.h"

#define A(r, c) A[N*(c) + (r)]

namespace
{

int alloc(int n, double** A, double** b)
{
	int N = n;
	N = (N + colAlign - 1)/colAlign*colAlign;
	*A = new double[N*n + N];
	*b = *A + N*n;
	return N;
}

void initFromInput(int n, int N, double* A, double* b)
{
	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i)
		{
			int res = scanf("%lf", A + N*j + i);
			(void)res;
		}

	for (int i = 0; i < n; ++i)
	{
		int res = scanf("%lf", b + i);
		(void)res;
	}
}

void initRandom(int n, int N, double* A, double* b)
{
	srand (time(NULL));
	const double scale = 100./RAND_MAX;

	for (int j = 0; j < n; ++j)
	{
		double sum = 0;
	
		for (int i = j + 1; i < n; ++i)
		{
			sum += A[N*j + i] = A[N*i + j] = rand()*scale;
		}
	
		A[N*j + j] = 2*sum + 1;
		b[j] = rand()*scale;
	}
}

} // namespace

int init(int argc, char** argv, int* n, double** A, double** b)
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

void printResult(int n, double* x, long long ns, double flopCount)
{
	for (int i = 0; i < n; ++i)
	    printf("%f\n", x[i]);
	    
	double giga = 1000*1000*1000;
	double seconds = ns/giga;
	fprintf(stderr, "Seconds elapsed: %f\nGFLOPS: %f\n", seconds, flopCount/giga/seconds);
}

