// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

using complex = thrust::complex<double>;

namespace ehrlich_aberth_jax {

namespace {

// CUDA kernel
__global__ void ehrlich_aberth_kernel(const int N, const int deg, const int itmax,
                                      const complex *coeffs, complex *roots, double *alpha,
                                      bool *conv, point *points, point *hull) {
  // Compute roots
  // This is a "grid-stride loop" see
  // http://alexminnaar.com/2019/08/02/grid-stride-loops.html

  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    ehrlich_aberth(deg, itmax, coeffs + tid * (deg + 1), roots + tid * deg,
                   alpha + tid * (deg + 1), conv + tid * deg, points + tid * (deg + 1),
                   hull + tid * (deg + 1));
    //    ehrlich_aberth_comp(deg, itmax, coeffs + tid * (deg + 1), roots + tid * deg,
    //                        alpha + tid * (deg + 1), conv + tid * deg, points + tid * (deg + 1),
    //                        hull + tid * (deg + 1));
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
  const int N = d.size;
  const int deg = d.deg;
  const int itmax = 50;

  const complex *coeffs = reinterpret_cast<const complex *>(buffers[0]);
  complex *roots = reinterpret_cast<complex *>(buffers[1]);

  // Preallocate memory for temporary arrays used within the kernel allocating these
  // arrays within the kernel with `new` results in a an illegal memory access
  // error for some reason I don't understand
  double *alpha;
  bool *conv;
  point *points;
  point *hull;

  cudaMalloc(&alpha, N * (deg + 1) * sizeof(double));
  cudaMalloc(&conv, N * deg * sizeof(bool));
  cudaMalloc(&points, N * (deg + 1) * sizeof(point));
  cudaMalloc(&hull, N * (deg + 1) * sizeof(point));

  const int block_dim = 512;
  const int grid_dim = std::min<int>(1024, (N + block_dim - 1) / block_dim);

  ehrlich_aberth_kernel<<<grid_dim, block_dim, 0, stream>>>(N, deg, itmax, coeffs, roots, alpha,
                                                            conv, points, hull);

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
