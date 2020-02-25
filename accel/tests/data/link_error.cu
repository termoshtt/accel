__device__ int f(int a);  // will cause undefined reference

__global__ void add(const int a[], const int b[], int c[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = f(a[i] + b[i]);
}

