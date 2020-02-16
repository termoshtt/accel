__device__ int f1(int a);

__global__ void add_and_t(const int a[], const int b[], int c[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = f1(a[i] + b[i]);
}
