// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace ehrlich_aberth_jax {

namespace {

__global__ void ehrlich_aberth_kernel(std::int64_t size, std::int64_t deg,
                                      const thrust::complex<double> *poly,
                                      thrust::complex<double> *roots, double *alpha, bool *conv,
                                      point *points, point *hull) {
  const std::int64_t itmax = 50;

  // Compute roots
  // This is a "grid-stride loop" see http://alexminnaar.com/2019/08/02/grid-stride-loops.html
  std::int64_t i;
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    i = idx * (deg + 1);
    ehrlich_aberth(poly + i, roots + i - idx, deg, itmax, alpha + i, conv + i - idx, points + i,
                   hull + i);
  }
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

inline void apply_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                                 std::size_t opaque_len) {
  const EhrlichAberthDescriptor &d =
      *UnpackDescriptor<EhrlichAberthDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;
  const std::int64_t deg = d.deg;

  const thrust::complex<double> *poly =
      reinterpret_cast<const thrust::complex<double> *>(buffers[0]);
  thrust::complex<double> *roots = reinterpret_cast<thrust::complex<double> *>(buffers[1]);

  const int block_dim = 512;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);

  // Preallocate memory for temporary arrays used within the kernel allocating these
  // arrays within the kernel with `new` results in a an illegal memory access
  // error for some reason I don't understand
  double *alpha;
  bool *conv;
  point *points;
  point *hull;

  cudaMalloc(&alpha, size * (deg + 1) * sizeof(double));
  cudaMalloc(&conv, size * deg * sizeof(bool));
  cudaMalloc(&points, size * (deg + 1) * sizeof(point));
  cudaMalloc(&hull, size * (deg + 1) * sizeof(point));

  ehrlich_aberth_kernel<<<grid_dim, block_dim, 0, stream>>>(size, deg, poly, roots, alpha, conv,
                                                            points, hull);

  // Free memory
  cudaFree(alpha);
  cudaFree(conv);
  cudaFree(points);
  cudaFree(hull);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
  apply_ehrlich_aberth(stream, buffers, opaque, opaque_len);
}

}  // namespace ehrlich_aberth_jax
