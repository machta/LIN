/*  Short job 1
*/ 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>


void testuj(float *mX,float *mX2,int n,int m)
{
  // test spravnosti reseni
  // test for correctness
  int i,j,k,pk,err;
  float pom,k1;
  err=0;
  for(i=0;i<(n*m);i++)
  {    
    if (fabs(mX[i]-mX2[i])>0.1)  
    {
      err++;
      printf("%i = %g,%g\n",i,mX[i],mX2[i]);
    }
  }
  if (err!=0) printf("total ERR=%i/%i \n",err,(n*m));
}
 
  
int vyprazdni(int *temp,int k)
{
  // vyprazdni cache
  // flush cache
  int i,s=0;
  for(i=0;i<k;i++) s+=temp[i];
  return s;
}  


//!! zacatek casti k modifikaci
//!! beginning of part for modification
void Gauss_BS(const float* const __restrict__ inA, const float* const __restrict__ inB,
	float* const __restrict__ outX, int n, int m)
{
	/*
	// first version
	#pragma omp parallel for
	for(int k = 0; k < m; k++)
	{
		for(int i = n - 1; i >= 0; i--)
		{
			float s = inB[k + i*m];
			for(int j = i + 1; j < n; j++)
			{
				s -= inA[i*n + j]*outX[k + j*m];
			} 
			outX[k + i*m] = s/inA[i*n + i];
		}
	}*/
	
	/*
	// version with transposition
	#pragma omp parallel
	{
		float* tmpX = new float[n];
		
		#pragma omp for
		for(int k = 0; k < m; k++)
		{
			for(int i = n - 1; i >= 0; i--)
			{
				float s = inB[k + i*m];
				for(int j = i + 1; j < n; j++)
				{
					s -= inA[i*n + j]*tmpX[j];
				} 
				outX[k + i*m] = tmpX[i] = s/inA[i*n + i];
			}
		}
		
		delete[] tmpX;
	}*/
	
	/*
	// simplified version with transposition
	#pragma omp parallel for
	for(int k = 0; k < m; k++)
	{
		float* tmpX = new float[n];
	
		for(int i = n - 1; i >= 0; i--)
		{
			float s = inB[k + i*m];
			for(int j = i + 1; j < n; j++)
			{
				s -= inA[i*n + j]*tmpX[j];
			} 
			outX[k + i*m] = tmpX[i] = s/inA[i*n + i];
		}
		
		delete[] tmpX;
	}*/

	
	/*// simplified version with transposition and optimized allocation
	float* X = new float[n*omp_get_max_threads()];
	
	#pragma omp parallel for
	for(int k = 0; k < m; k++)
	{
		float* tmpX = X + n*omp_get_thread_num();
	
		for(int i = n - 1; i >= 0; i--)
		{
			float s = inB[k + i*m];
			for(int j = i + 1; j < n; j++)
			{
				s -= inA[i*n + j]*tmpX[j];
			} 
			outX[k + i*m] = tmpX[i] = s/inA[i*n + i];
		}
	}
	
	delete[] X;*/
	
	// simplified version with transposition and optimized allocation #2
	float** X = new float*[omp_get_max_threads()];
	for (int i = 0; i < omp_get_max_threads(); i++)
		X[i] = new float[n];
	
	#pragma omp parallel for
	for(int k = 0; k < m; k++)
	{
		float* tmpX = X[omp_get_thread_num()];
	
		for(int i = n - 1; i >= 0; i--)
		{
			float s = inB[k + i*m];
			for(int j = i + 1; j < n; j++)
			{
				s -= inA[i*n + j]*tmpX[j];
			} 
			outX[k + i*m] = tmpX[i] = s/inA[i*n + i];
		}
	}
	
	for (int i = 0; i < omp_get_max_threads(); i++)
		delete[] X[i];
	delete[] X;
	
	/*
	// weird version
	int gcd;
	{
		int a = m, b = omp_get_max_threads(), tmp;
		while (b != 0)
		{
			tmp = a % b;

			a = b;
			b = tmp;
		}
		gcd = a;
	}
	//printf("gcd = %d\n", gcd);
	
	#pragma omp parallel num_threads(gcd)
	{
		float* tmpX = new float[n];
		
		#pragma omp for
		for(int k = 0; k < m; k++)
		{
			for(int i = n - 1; i >= 0; i--)
			{
				float s = inB[k + i*m];
				#pragma omp parallel for reduction(-:s) num_threads(omp_get_max_threads()/gcd)
				for(int j = i + 1; j < n; j++)
				{
					s -= inA[i*n + j]*tmpX[j];
				} 
				outX[k + i*m] = tmpX[i] = s/inA[i*n + i];
			}
		}
		
		delete[] tmpX;
	}*/
}
//!! end of part for modification
//!! konec casti k modifikaci



int main( void ) {

 double start_time,end_time,timea[10];
 
 int soucet=0,N,i,j,k,n,m,*pomo,v;
 int ri,rj,rk;
 double delta,s_delta=0.0;
 float *mA, *mB,*mX,*mX2,s; 
    
  //int tn[4]={1000,1500,2000,2500};  
  int tn[4]={1*1024,2*1024,12*1024,16*1024};
  int tm[4]={1024,256,32,12};
  srand (time(NULL));   
  pomo=(int *)malloc(32*1024*1024);    
  v=0;    
  
  for(N=0;N<10;N++) timea[N]=0.0;
  
  for(N=0;N<4;N++)
  {
  n=tn[N];
  m=tm[N];

  mA=(float *)malloc(n * n * sizeof(float));
  mB=(float *)malloc(n * m * sizeof(float));
  mX=(float *)malloc(n * m *sizeof(float));
  mX2=(float *)malloc(n * m * sizeof(float));
  if ((mA==NULL)||(mB==NULL)||(mX==NULL)||(mX2==NULL)) 
  {
    printf("Insufficient memory!\n"); 
    return -1;
  }
  
  for (i=0; i<n; i++) {
  for (j=0; j<n; j++) {
    if (j>=i)
      mA[i*n+j] = (float)(2*(rand()%59)-59);
    else  
      mA[i*n+j] = 0.0;
  }}
  for (k=0; k<m; k++) {
  for (j=0; j<n; j++) {  
  mX2[j*m+k] = (float)((rand()%29)-14);     
  }}
  
  for (k=0; k<m; k++) {
  for (i=0; i<n; i++) {
  s=0.0;
  for (j=0; j<n; j++) {
    s += mA[i*n+j]*(mX2[j*m+k]);
  }
  mB[i*m+k]=s;
  }}
  soucet+=vyprazdni(pomo,v);
  start_time=omp_get_wtime();
  // improve performance of this call
  // vylepsit vykonnost tohoto volani
  Gauss_BS( mA, mB, mX, n,m);
                          
  end_time=omp_get_wtime();
  delta=end_time-start_time;
  timea[0]+=delta;
  testuj(mX,mX2,n,m);
  printf("n0=%i m0=%i time=%g \n",n,m,delta);        
  fflush(stdout);
 
  free(mX2);
  free(mX);
  free(mB);
  free(mA);
  } 
  printf("%i\n",soucet); 
  printf("Total time=%g\n",timea[0]);
  //for(N=0;N<10;N++)  printf("%i=%g\n",N,timea[N]); 

  return 0;
}
