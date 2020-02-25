__global__ void add(const int a[], const int b[], int c[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
