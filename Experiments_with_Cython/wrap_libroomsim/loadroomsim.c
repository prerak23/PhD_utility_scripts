#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "loadroomsim.h"



double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}
double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

BRIR *populate(double fs, int nChannels, int nSamples)
{
	int nSources = 1;
	int nReceivers = 2;
	double sigma = 82.0;
  double Mi = 40.0;
	BRIR *br=(BRIR *)calloc(nSources * nReceivers + 1, sizeof(BRIR));
	int k,j,l;
	br[0].fs = fs;
	br[0].nChannels = nChannels;
	br[0].nSamples = nSamples;
	br[0].sample = (double *)calloc(nChannels * nSamples, sizeof(double));

  br[1].fs = fs;
	br[1].nChannels = nChannels;
	br[1].nSamples = nSamples;
	br[1].sample = (double *)calloc(nChannels * nSamples, sizeof(double));

for (l=0;l<nReceivers;l++)
{
	for(k=0;k<nChannels;k++)
	{
		for(j=0;j<nSamples;j++)
		{
			br[l].sample[k*nSamples+j]=normalRandom()*sigma+Mi;
		}
	}
}
	return br;
}
