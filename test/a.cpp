#include<iostream>
#define SIZE 10000000

int main(){
	double* x = (double*)malloc(sizeof(double) * SIZE);
	for(int i=0; i<SIZE; i++){
		x[i] = 1.0;
	}

	double goma = -12345;
	int i = 0;

// #pragma acc data copy(x[0:SIZE]) 
// 	{
// #pragma acc host_data use_device(x)
// 		{
// 			goma = cublasDasum(SIZE, x, 1);
// 		}
// 	}
	for(int c = 0; c < 1000; c++){
#pragma acc data copy(x[0:SIZE])
		{
#pragma acc kernels
			{
#pragma acc loop independent
				for(i=0; i<SIZE; i++){
					goma += x[i];
				}
			}
		}
	}

	std::cout << "goma = " << goma << std::endl;
	return 0;
}
// 		cublasInit();
// 		goma = cublasDasum(SIZE, x, 1);
// 		cublasShutdown();
