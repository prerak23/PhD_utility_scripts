typedef struct {
	double fs;
	int nChannels;
	int nSamples;
	double *sample;
} BRIR;


BRIR *populate(double, int , int );
