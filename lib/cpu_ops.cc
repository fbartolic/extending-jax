// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "ehrlich_aberth.h"
#include "pybind11_kernel_helpers.h"

using namespace ehrlich_aberth;

namespace {

void cpu_ehrlich_aberth(void *out_vector, const void **in) {
  // Parse the inputs
  // reinterpret_cast here converts a void* to a pointer to a pointer for a specific type
  const std::int64_t size =
      *reinterpret_cast<const std::int64_t *>(in[0]);  // number of polynomials (size of problem)
  const std::int64_t deg =
      *reinterpret_cast<const std::int64_t *>(in[1]);  // degree of polynomials
  const std::int64_t itmax = *reinterpret_cast<const std::int64_t *>(in[2]);  // maxiter

  // Flattened polynomial coefficients, shape (deg + 1)*size
  const std::complex<double> *poly_flattened =
      reinterpret_cast<const std::complex<double> *>(in[3]);

  // Output roots, shape deg*size
  std::complex<double> *roots = reinterpret_cast<std::complex<double> *>(out);

  // Compute roots
  std::int64_t j = 0;
  for (std::int64_t i = 0; i < size * deg; i += (deg + 1)) {
    ehrlich_aberth::ehrlich_aberth(poly_flattened + i, roots + i - j, deg, itmax);
    j++;
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_ehrlich_aberth"] = EncapsulateFunction(cpu_ehrlich_aberth<std::complex<double>>);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
