CXX=g++-9
CXX_FLAGS=-O3 -std=c++11 
ACC_FLAGS=-fopenacc -foffload=nvptx-none="-misa=sm_35" -lcuda

FC=gfortran-9
FC_FLAGS=-O3 -std=f2018

TARGET=acc_naked.out acc_cublas.out acc_fortran.out

ACC_INCLUDE=-I/usr/local/cuda-11.0/targets/x86_64-linux/include/
ACC_INCLUDE+=-I/usr/local/cuda-10.0/targets/x86_64-linux/include/ 
ACC_LIB=-L/usr/local/cuda-11.0/targets/x86_64-linux/lib/
ACC_LIB+=-L/usr/local/cuda-10.0/targets/x86_64-linux/lib/ 
ACC_LIBS=-lcuda -lcublas -lcudart

all: $(TARGET)

acc_naked.out: acc_naked.cpp
	$(CXX) $(ACC_FLAGS) $(CXX_FLAGS) $< -o $@

acc_cublas.out: acc_cublas.cpp
	$(CXX) $(ACC_FLAGS) $(CXX_FLAGS) $(ACC_INCLUDE) $(ACC_LIB) $< -o $@ $(ACC_LIBS)

acc_fortran.out: acc_fortran.f90
	$(FC) $(ACC_FLAGS) $(FC_FLAGS) $< -o $@

test: $(TARGET)
	./acc_naked.out 1000000
	./acc_cublas.out 1000000
	nvprof ./acc_cublas.out 1000000
	./acc_fortran.out 1000000

clean:
	- rm *.out
