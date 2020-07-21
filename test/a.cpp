#include<iostream>
#include<cstdlib>

int main(int argc, char** argv){

	int size = 0;
	if(argc == 2){
		size = std::atoi(argv[1]);
	
	}
	else{
		std::cout << "error, $1 is vector size" << std::endl;
		return 1;
	}

	double* x = (double*)malloc(sizeof(double) * size);
	double* y = (double*)malloc(sizeof(double) * size);

	for(int i=0; i<size; i++){
		x[i] = 1.0;
		y[i] = 2.0;
	}

	double dot = 0.0;

#pragma acc kernels copy(x[0:size], y[0:size])
	{
			for(int i=0; i<size; i++){
				dot += x[i] * y[i];
			}
	}

	if(dot != 2.0 * size){
		std::cout << "dot = " << dot << std::endl;
		std::cout << "error!" << std::endl;
		return 1;
	}
	else{
		std::cout << "dot = " << dot << std::endl;
		std::cout << "Pass!" << std::endl;
		return 0;
	}
}
