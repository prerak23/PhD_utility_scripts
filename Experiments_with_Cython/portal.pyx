# distutils: sources = dm.c
# distutils: include_dirs = /home/psrivast/baseline/scripts/Cython


cimport ctest

cdef unsigned long abc

ctest.getSeconds(&abc)
print(abc)

