cdef struct BRIR:
   double fs
   int channels
   int nSamples
   double* sample

cdef extern from "loadroomsim.h":
          BRIR *populate(double, int, int);
