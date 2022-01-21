// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace ehrlich_aberth_jax {

namespace {

__global__ void ehrlich_aberth_kernel(std::int64_t size, std::int64_t deg,
                                      const std::complex<double> *poly,
                                      std::complex<double> *roots) {
  const std::int64_t itmax = 50;

  // Compute roots
  std::int64_t i;
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    i = idx * (deg + 1);
    ehrlich_aberth::ehrlich_aberth(pooly + i, roots + i - idx, deg, itmax);
  }
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

  const std::complex<double> *poly = reinterpret_cast<const std::complex<double> *>(buffers[0]);
  std::complex<double> *roots = reinterpret_cast<std::complex<double> *>(buffers[1]);

  const int block_dim = 128;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
  ehrlich_aberth_kernel<std::complex<double>>
      <<<grid_dim, block_dim, 0, stream>>>(size, deg, poly, roots);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
  apply_ehrlich_aberth<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace ehrlich_aberth_jax