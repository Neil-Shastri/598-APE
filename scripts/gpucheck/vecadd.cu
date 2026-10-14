#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 24;
  size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  cudaMalloc(&a, bytes);
  cudaMalloc(&b, bytes);
  cudaMalloc(&c, bytes);
  cudaMemset(a, 0, bytes);
  cudaMemset(b, 0, bytes);
  vecadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaError_t err = cudaDeviceSynchronize();
  printf("vecadd: %s\n", cudaGetErrorString(err));
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return err != cudaSuccess;
}
