
# distutils: sources = loadroomsim.c
# distutils: include_dirs = /home/psrivast/baseline/scripts/Cython/wrap_libroomsim

cimport cload_test
import numpy as np

cpdef response_brir():
    cdef int nSamp = int(1.25*44100)
    ad=cload_test.populate(44100.00, 2, nSamp)
    n=0
    nsrc=1
    nrec=2
    for k in range(nsrc*nrec):

        chan=int(ad[k].channels)
        sam=int(ad[k].nSamples)
        print(chan*sam)
        fs = ad[k].fs

        np_arr=np.empty([chan,sam],dtype=float)
        j=0
        for i in range(chan*sam):
            if i >= sam:
                j=1
                np_arr[j][i-sam]=float(ad[k].sample[i])
            else:
                np_arr[j][i]=float(ad[k].sample[i])

        print(np_arr.shape)
