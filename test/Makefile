CXX=g++-8
CXX_FLAGS=-O3 -std=c++11 
ACC_FLAGS=-fopenacc -foffload=nvptx-none -lcuda

TARGET=acc_naked.out acc_cublas.out

ACC_INCLUDE=-I/usr/local/cuda-10.0/targets/x86_64-linux/include/ 
ACC_LIB=-L/usr/local/cuda-10.0/targets/x86_64-linux/lib/ 
ACC_LIBS=-lcuda -lcublas -lcudart

all: $(TARGET)

acc_naked.out: acc_naked.cpp
	$(CXX) $(ACC_FLAGS) $(CXX_FLAGS) $< -o $@

acc_cublas.out: acc_cublas.cpp
	$(CXX) $(ACC_FLAGS) $(CXX_FLAGS) $(ACC_INCLUDE) $(ACC_LIB) $< -o $@ $(ACC_LIBS)

test:
	./acc_naked.out 1000000
	./acc_cublas.out 1000000
	nvprof ./acc_cublas.out 1000000

clean:
	- rm *.out
