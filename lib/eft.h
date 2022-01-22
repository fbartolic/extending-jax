#ifndef EFT
#define EFT
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/complex.h>

namespace ehrlich_aberth_jax {

#ifdef __CUDACC__
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE inline
#endif

/* EFT Data Structure */
struct eft {
  double fl_res, fl_err;
};
/* EFT Cmplx Data Structure for Sum */
struct eft_cmplx_sum {
  thrust::complex<double> fl_res, fl_err;
};
/* EFT Cmplx Data Structure for Product */
struct eft_cmplx_prod {
  thrust::complex<double> fl_res, fl_err1, fl_err2, fl_err3;
};
}  // namespace ehrlich_aberth_jax
#endif