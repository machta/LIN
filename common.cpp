#include "common.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <omp.h>

using namespace std;

namespace
{

void setMatrix(real val, int r, int c, int n, int k, real* A)
{
	column2Tiled(&r, &c, n, k);
	A[n*c + r] = val;	
}

void alloc(int n, real** A, real** b)
{
	const int colAlign = 4;	(void)colAlign;
	
	*A = new real[n*n + n];
	*b = *A + n*n;
}

// Load the values from stdin.
void initFromInput(int n, int k, real* A, real* b)
{
	float tmp;

	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i)
		{
			int res = scanf("%f", &tmp);
			setMatrix(tmp, i, j, n, k, A);
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
void initRandom(int n, int k, real* A, real* b)
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
			double tmp = rand()*scale;
			columnSums[j] += columnSums[i] += tmp;
			setMatrix(tmp, i, j, n, k, A);
			setMatrix(tmp, j, i, n, k, A);
		}
		
		setMatrix(2*columnSums[j] + 1, j, j, n, k, A);
		b[j] = rand()*scale;
	}
	
	delete[] columnSums;
}

} // namespace

int init(int argc, char** argv, int* n, real** A, real** b, bool tiled)
{
	if (argc < 2)
		error("not enough parameters.");
		
	int k = 0;
	
	if (argv[1][0] == 'i')
	{
		int res = scanf("%d", n);
		(void)res;
		
		k = *n;
		if (argc > 2)
		{
			k = stoi(argv[2]);
		}
		
		alloc(*n, A, b);
		initFromInput(*n, tiled ? k : *n, *A, *b);
	}
	else if (argv[1][0] == 'r')
	{
		if (argc <= 2)
			error("not enough parameters.");
			
		*n = stoi(argv[2]);
		
		k = *n;
		if (argc > 3)
		{
			k = stoi(argv[3]);
		}
		
		alloc(*n, A, b);
		initRandom(*n, tiled ? k : *n, *A, *b);
	}
	else
	{
		error("only 'r' and 'i' modes are allowed.");
	}
	
	return k;
}

void printResult(int n, real* x, long long ns, double flopCount, int k)
{
	for (int i = 0; i < n; ++i)
	    printf("%.60f\n", x[i]);
	    
	double giga = 1000*1000*1000;
	double seconds = ns/giga;
	
	fprintf(stderr, "#   n block thread         seconds          GFLOPS\n");
	fprintf(stderr, "%5d %5d %6d %15f %15f\n", n, k, omp_get_max_threads(), seconds, flopCount/giga/seconds);
}

void column2Tiled(int *r, int *c, int n, int k)
{
	int blockR = *r/k;
	int blockC = *c/k;
	int blockH = min(k, n - blockR*k);
	int blockW = min(k, n - blockC*k);
	int blockStart = n*k*blockC + k*blockW*blockR;
	int i = blockStart + blockH*(*c - blockC*k) + *r - blockR*k;
	
	//fprintf(stderr, "r = %2d, c = %2d, blockR = %d, blockC = %d, i = %2d\n", r, c, blockR, blockC, i);
	
	*c = i/n;
	*r = i - i/n*n;
}

void printMatrix(int n, int k, real* A)
{
	using namespace std;
	
	real* max = max_element(A, A + n*n, [] (real a, real b) { return fabs(a) < fabs(b); });
	int digits = std::max<double>(1., log(fabs(*max))) + 5;
	
	stringstream ss;
	ss << "%" << digits << ".2f";
	
	for (int i = 0; i < n; ++i)
	{
        for (int j = 0; j < n; ++j)
        {
        	int r = i, c = j;
        	column2Tiled(&r, &c, n, k);
        	fprintf(stderr, ss.str().c_str(), A[n*c + r]);
        }
        fprintf(stderr, "\n");
	}
}
