#ifndef EHRLICH_ABERTH
#define EHRLICH_ABERTH
#include <cstdbool>
#include <cstdio>

#include "horner.h"
#include "init_est.h"

namespace ehrlich_aberth_jax {

#ifdef __CUDACC__
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE inline
#endif

/* point convergence structure */
typedef struct {
  bool x, y;
} point_conv;
/* ehrlich aberth correction term */
thrust::complex<double> correction(const thrust::complex<double> *roots,
                                   const thrust::complex<double> h,
                                   const thrust::complex<double> hd, const unsigned int deg,
                                   const unsigned int j) {
  thrust::complex<double> corr = 0;
  for (int i = 0; i < j; i++) {
    corr += 1. / (roots[j] - roots[i]);
  }
  for (int i = j + 1; i < deg; i++) {
    corr += 1. / (roots[j] - roots[i]);
  }
  return h / (hd - h * corr);
}
/* reverse ehrlich aberth correction term */
thrust::complex<double> rcorrection(const thrust::complex<double> *roots,
                                    const thrust::complex<double> h,
                                    const thrust::complex<double> hd, const unsigned int deg,
                                    const unsigned int j) {
  thrust::complex<double> corr = 0;
  for (int i = 0; i < j; i++) {
    corr += 1. / (roots[j] - roots[i]);
  }
  for (int i = j + 1; i < deg; i++) {
    corr += 1. / (roots[j] - roots[i]);
  }
  return thrust::pow(roots[j], 2) * h /
         ((double)deg * roots[j] * h - hd - thrust::pow(roots[j], 2) * h * corr);
}
/* The function we want to expose as a JAX primitive*/
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void ehrlich_aberth(const thrust::complex<double> *poly,
                                                        thrust::complex<double> *roots,
                                                        const unsigned int deg,
                                                        const unsigned int itmax) {
  // local variables
  bool s;
  double b;
  thrust::complex<double> h, hd;
  // local arrays
  double alpha[deg + 1];
  bool conv[deg];
  // initial estimates
  for (int i = 0; i < deg; i++) {
    alpha[i] = thrust::abs(poly[i]);
    conv[i] = false;
  }
  alpha[deg] = thrust::abs(poly[deg]);
  init_est(alpha, deg, roots);
  // update initial estimates
  for (int i = 0; i <= deg; i++) {
    alpha[i] = alpha[i] * fma(3.8284271247461900976, i, 1);
  }
  for (int i = 0; i < itmax; i++) {
    for (int j = 0; j < deg; j++) {
      if (conv[j] == 0) {
        if (thrust::abs(roots[j]) > 1) {
          rhorner_dble(alpha, 1. / thrust::abs(roots[j]), deg, &b);
          rhorner_cmplx(poly, 1. / roots[j], deg, &h, &hd);
          if (thrust::abs(h) > EPS * b) {
            roots[j] = roots[j] - rcorrection(roots, h, hd, deg, j);
          } else {
            conv[j] = true;
          }
        } else {
          horner_dble(alpha, thrust::abs(roots[j]), deg, &b);
          horner_cmplx(poly, roots[j], deg, &h, &hd);
          if (thrust::abs(h) > EPS * b) {
            roots[j] = roots[j] - correction(roots, h, hd, deg, j);
          } else {
            conv[j] = true;
          }
        }
      }
    }
    s = conv[0];
    for (int j = 1; j < deg; j++) {
      s = s && conv[j];
    }
    if (s) {
      break;
    }
  }
  if (!s) {
    printf("not all roots converged\n");
  }
}
///* ehrlich_aberth with compensated arithmetic*/
// void ehrlich_aberth_comp(const thrust::complex<double> *poly, thrust::complex<double> *roots,
// const unsigned int deg,
//                         const unsigned int itmax) {
//  // local variables
//  int s;
//  double b;
//  thrust::complex<double> h, hd;
//  // local arrays
//  double alpha[deg + 1];
//  point_conv conv[deg];
//  // initial estimates
//  for (int i = 0; i < deg; i++) {
//    alpha[i] = thrust::abs(poly[i]);
//    conv[i].x = false;
//    conv[i].y = false;
//  }
//  alpha[deg] = thrust::abs(poly[deg]);
//  init_est(alpha, deg, roots);
//  // update initial estimates
//  for (int i = 0; i <= deg; i++) {
//    alpha[i] = alpha[i] * fma(3.8284271247461900976, i, 1);
//  }
//  for (int i = 0; i < itmax; i++) {
//    for (int j = 0; j < deg; j++) {
//      if (!conv[j].x) {
//        if (thrust::abs(roots[j]) > 1) {
//          rhorner_dble(alpha, 1. / thrust::abs(roots[j]), deg, &b);
//          rhorner_cmplx(poly, 1. / roots[j], deg, &h, &hd);
//          if (thrust::abs(h) > EPS * b) {
//            roots[j] = roots[j] - rcorrection(roots, h, hd, deg, j);
//          } else {
//            conv[j].x = true;
//          }
//        } else {
//          horner_dble(alpha, thrust::abs(roots[j]), deg, &b);
//          horner_cmplx(poly, roots[j], deg, &h, &hd);
//          if (thrust::abs(h) > EPS * b) {
//            roots[j] = roots[j] - correction(roots, h, hd, deg, j);
//          } else {
//            conv[j].x = true;
//          }
//        }
//      } else if (!conv[j].y) {
//        if (thrust::abs(roots[j]) > 1) {
//          rhorner_comp_cmplx(poly, 1. / roots[j], deg, &h, &hd, &b);
//          double errBound =
//              EPS * thrust::abs(h) + (gamma_const(4 * deg + 2) * b + 2 * pow(EPS, 2) *
//              thrust::abs(h));
//          if (thrust::abs(h) > 4 * errBound) {
//            thrust::complex<double> corr = rcorrection(roots, h, hd, deg, j);
//            if (thrust::abs(corr) > 4 * EPS * thrust::abs(roots[j])) {
//              roots[j] = roots[j] - corr;
//            } else {
//              conv[j].y = true;
//            }
//          } else {
//            conv[j].y = true;
//          }
//        } else {
//          horner_comp_cmplx(poly, roots[j], deg, &h, &hd, &b);
//          double errBound =
//              EPS * thrust::abs(h) + (gamma_const(4 * deg + 2) * b + 2 * pow(EPS, 2) *
//              thrust::abs(h));
//          if (thrust::abs(h) > 4 * errBound) {
//            thrust::complex<double> corr = correction(roots, h, hd, deg, j);
//            if (thrust::abs(corr) > 4 * EPS) {
//              roots[j] = roots[j] - corr;
//            } else {
//              conv[j].y = true;
//            }
//          } else {
//            conv[j].y = true;
//          }
//        }
//      }
//    }
//    s = conv[0].y;
//    for (int j = 1; j < deg; j++) {
//      s = s && conv[j].y;
//    }
//    if (s) {
//      break;
//    }
//  }
//  if (!s) {
//    printf("not all roots comp converged\n");
//  }
//}
}  // namespace ehrlich_aberth_jax
#endif
