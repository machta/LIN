#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
	double n;
	
	vector<double> x;
	
	while (scanf("%lf", &n) == 1 && feof(stdin) == 0)
	{
		x.push_back(n);
	}
	x.push_back(0);
	
	double maxX = 0, maxDiff = 0;
	int half = x.size()/2;
	
	for (int i = 0; i < half; ++i)
	{
		fprintf(stderr, "%10f %10f\n", x[i], abs(x[i] - x[i + half]));
	
		maxX = max(maxX, abs(x[i]));
		maxDiff = max(maxDiff, abs(x[i] - x[i + half]));
	}
	
	printf("%f\n", maxDiff/maxX);
	
	return 0;
}
