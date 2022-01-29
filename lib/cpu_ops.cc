// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "ehrlich_aberth.h"
#include "pybind11_kernel_helpers.h"

using complex = thrust::complex<double>;

using namespace ehrlich_aberth_jax;

namespace {

void cpu_ehrlich_aberth(void *out, const void **in) {
  // Parse the inputs
  // reinterpret_cast here converts a void* to a pointer to a pointer for a specific type
  const int size =
      *reinterpret_cast<const int *>(in[0]);  // number of polynomials (size of problem)
  const int deg = *reinterpret_cast<const int *>(in[1]);  // degree of polynomials
  //  const int itmax = *reinterpret_cast<const int *>(in[2]);  // maxiter

  const int itmax = 50;

  // Flattened polynomial coefficients, shape (deg + 1)*size
  const complex *coeffs = reinterpret_cast<const complex *>(in[2]);

  // Output roots, shape deg*size
  complex *roots = reinterpret_cast<complex *>(out);

  // Allocate memory for temporary arrays
  double *alpha = new double[deg + 1];
  bool *conv = new bool[deg];
  point *points = new point[deg + 1];
  point *hull = new point[deg + 1];

  // Compute roots
  for (int idx = 0; idx < size; ++idx) {
    ehrlich_aberth_jax::ehrlich_aberth(deg, itmax, coeffs + idx * (deg + 1), roots + idx * deg,
                                       alpha, conv, points, hull);
  }

  // Free memory
  delete[] alpha;
  delete[] conv;
  delete[] points;
  delete[] hull;
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_ehrlich_aberth"] = EncapsulateFunction(cpu_ehrlich_aberth);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
