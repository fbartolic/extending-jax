#ifndef HORNER
#define HORNER
#include <cfloat>
#include <cmath>
#include <cstdio>

#include "eft.h"
namespace ehrlich_aberth_jax {

/* Global Constants */
const double EPS = DBL_EPSILON / 2;
const double ETA = DBL_MIN;
/* Unit in First Place */
double ufp(const double p) {
  double q = p / DBL_EPSILON + p;
  return fabs(fma(q, EPS - 1, q));
}
/* Gamma Constant */
double gamma_const(const unsigned int n) {
  double s = 1.41421356237309504880;
  double g = (2 * n * EPS) * s;
  return g / ((1 - DBL_EPSILON) - g);
}
/* Fast Two Sum */
void fast_two_sum(const double a, const double b, struct eft *res) {
  res->fl_res = a + b;
  res->fl_err = (a - res->fl_res) + b;
}
/* Two Sum */
void two_sum(const double a, const double b, struct eft *res) {
  res->fl_res = a + b;
  double t = res->fl_res - a;
  res->fl_err = (a - (res->fl_res - t)) + (b - t);
}
/* Two Product */
void two_prod(const double a, const double b, struct eft *res) {
  res->fl_res = a * b;
  res->fl_err = fma(a, b, -res->fl_res);
}
/* Two Sum Cmplx */
void two_sum_cmplx(const std::complex<double> a, const std::complex<double> b, struct eft *eft_arr,
                   struct eft_cmplx_sum *res) {
  // struct eft real_eft, imag_eft;
  two_sum(a.real(), b.real(), eft_arr);
  two_sum(a.imag(), b.imag(), eft_arr + 1);
  res->fl_res = std::complex<double>(eft_arr[0].fl_res, eft_arr[1].fl_res);
  res->fl_err = std::complex<double>(eft_arr[0].fl_err, eft_arr[1].fl_err);
}
/* Two Product Cmplx */
void two_prod_cmplx(const std::complex<double> a, const std::complex<double> b,
                    struct eft *eft_arr, struct eft_cmplx_prod *res) {
  // struct eft real_prod1, real_prod2, real_prod3, real_prod4, real_sum1, real_sum2;
  two_prod(a.real(), b.real(), eft_arr);
  two_prod(a.imag(), b.imag(), eft_arr + 1);
  two_prod(a.real(), b.imag(), eft_arr + 2);
  two_prod(a.imag(), b.real(), eft_arr + 3);
  two_sum(eft_arr[0].fl_res, -eft_arr[1].fl_res, eft_arr + 4);
  two_sum(eft_arr[2].fl_res, eft_arr[3].fl_res, eft_arr + 5);
  res->fl_res = std::complex<double>(eft_arr[4].fl_res, eft_arr[5].fl_res);
  res->fl_err1 = std::complex<double>(eft_arr[0].fl_err, eft_arr[2].fl_err);
  res->fl_err2 = std::complex<double>(-eft_arr[1].fl_err, eft_arr[3].fl_err);
  res->fl_err3 = std::complex<double>(eft_arr[4].fl_err, eft_arr[5].fl_err);
}
/* Error Free Array Extraction */
double extract(double *p, double sigma) {
  struct eft res;
  for (int i = 0; i < 4; ++i) {
    // two_sum(sigma,p[i],&res);
    fast_two_sum(sigma, p[i], &res);
    sigma = res.fl_res;
    p[i] = res.fl_err;
  }
  return sigma;
}
/* Sum */
double sum(const double *p) {
  double s = p[0];
  for (int i = 1; i < 4; ++i) {
    s += p[i];
  }
  return s;
}
/* Absolute Sum */
double abs_sum(const double *p) {
  double s = fabs(p[0]);
  for (int i = 1; i < 4; ++i) {
    s += fabs(p[i]);
  }
  return s;
}
/* Fast Accurate Summation */
double fast_acc_sum(double *p) {
  // variables
  int n = 0;
  double phi, sigma, sigma_new, T, t, t_new, tau, u;
// goto label
start:
  T = abs_sum(p) / fma(-4, EPS, 1);
  if (T <= ETA / EPS) {
    return sum(p);  // no rounding error
  }
  t = 0;
  do {
    sigma = (2 * T) / fma(-13, EPS, 1);  // 3 flops
    sigma_new = extract(p, sigma);       // 4(3) = 12 flops
    tau = sigma_new - sigma;             // 1 flop
    t_new = t;
    t = t_new + tau;  // 1 flop
    if (t == 0) {
      goto start;  // intermediate sum is zero, recursively apply fast_acc_sum to array of lower
                   // order parts
    }
    u = ufp(sigma);                                                   // 4 flops
    phi = ((48 * EPS) * u) / fma(-5, EPS, 1);                         // 4 flops
    T = fmin((fma(4, EPS, 1.5) * (4 * EPS)) * sigma, (8 * EPS) * u);  // 6 flops
    n += 1;
  } while ((fabs(t) < phi) && (4 * T > ETA / EPS));  // 2 flops
  tau = (t_new - t) + tau;
  // return
  return t + (tau + sum(p));
}
/* Fast Complex Accurate Summation */
std::complex<double> fast_cmplx_acc_sum(std::complex<double> *p) {
  // variables
  double realp[4] = {p[0].real(), p[1].real(), p[2].real(), p[3].real()};
  double imagp[4] = {p[0].imag(), p[1].imag(), p[2].imag(), p[3].imag()};
  // return
  return std::complex<double>(fast_acc_sum(realp), fast_acc_sum(imagp));
}
/* Sort */
void sort(double *p) {
  double max, temp;
  int ind, i, j;
  for (i = 0; i < 3; ++i) {
    max = fabs(p[i]);
    ind = i;
    for (j = i + 1; j < 4; ++j) {
      temp = fabs(p[j]);
      if (temp > max) {
        max = temp;
        ind = j;
      }
    }
    if (ind != i) {
      temp = p[i];
      p[i] = p[ind];
      p[ind] = temp;
    }
  }
}
/* Priest Summation */
double priest_sum(double *p) {
  // sort p
  sort(p);
  // initialize
  double s = p[0], c = 0, y, u, t, v, z;
  for (int i = 1; i < 4; ++i) {
    y = c + p[i];
    u = p[i] - (y - c);
    t = y + s;
    v = y - (t - s);
    z = u + v;
    s = t + z;
    c = z - (s - t);
  }
  // return
  return s;
}
/* Priest Complex Summation */
std::complex<double> priest_cmplx_sum(std::complex<double> *p) {
  // variables
  double realp[4] = {p[0].real(), p[1].real(), p[2].real(), p[3].real()};
  double imagp[4] = {p[0].imag(), p[1].imag(), p[2].imag(), p[3].imag()};
  // return
  return std::complex<double>(priest_sum(realp), priest_sum(imagp));
}
/* Horner Method with Double Real Arithmetic */
void horner_dble(const double *poly, const double x, const unsigned int deg, double *h) {
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    *h = fma(*h, x, poly[i]);
  }
}
/* Reversal Horner Method with Double Real Arithmetic */
void rhorner_dble(const double *poly, const double x, const unsigned int deg, double *h) {
  // Reversal Horner's method
  *h = poly[0];
  for (int i = 1; i <= deg; ++i) {
    *h = fma(*h, x, poly[i]);
  }
}
/* Horner Method with Complex Arithmetic */
void horner_cmplx(const std::complex<double> *poly, const std::complex<double> x,
                  const unsigned int deg, std::complex<double> *h, std::complex<double> *hd) {
  // Horner's method
  *h = poly[deg];
  *hd = 0;
  for (int i = deg - 1; i >= 0; --i) {
    *hd = *hd * x + *h;
    *h = *h * x + poly[i];
  }
}
/* Reversal Horner Method with Complex Arithmetic */
void rhorner_cmplx(const std::complex<double> *poly, const std::complex<double> x,
                   const unsigned int deg, std::complex<double> *h, std::complex<double> *hd) {
  // Reversal Horner's method
  *h = poly[0];
  *hd = 0;
  for (int i = 1; i <= deg; ++i) {
    *hd = *hd * x + *h;
    *h = *h * x + poly[i];
  }
}
/* Horner Method */
void horner(const std::complex<double> *poly, const std::complex<double> x, const unsigned int deg,
            std::complex<double> *h) {
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    *h = *h * x + poly[i];
  }
}
/* Horner's Method with Complex Compensated Arithmetic */
void horner_comp_cmplx(const std::complex<double> *poly, const std::complex<double> x,
                       const unsigned int deg, std::complex<double> *h, std::complex<double> *hd,
                       double *eb) {
  // local variables
  struct eft eft_arr[6];
  struct eft_cmplx_sum tsc;
  struct eft_cmplx_prod tpc;
  std::complex<double> e = 0, ed = 0;
  std::complex<double> p[4];
  double ap[4];
  // Horner's method
  *h = poly[deg];
  *hd = 0;
  *eb = 0;
  for (int i = deg - 1; i >= 0; --i) {
    // product and sum for derivative evaluation
    two_prod_cmplx(*hd, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, *h, eft_arr, &tsc);
    // update hd and ed
    *hd = tsc.fl_res;
    // ed = ed*x + e + (tpc.fl_err1 + tpc.fl_err2 + tpc.fl_err3 + tsc.fl_err);
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    ed = ed * x + e + priest_cmplx_sum(p);
    // product and sum for polynomial evaluation
    two_prod_cmplx(*h, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, poly[i], eft_arr, &tsc);
    // update h and e
    *h = tsc.fl_res;
    // e = e*x + (tpc.fl_err1 + tpc.fl_err2 + tpc.fl_err3 + tsc.fl_err);
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    e = e * x + priest_cmplx_sum(p);
    // update error bound
    // *eb = *eb*std::abs(x) + (std::abs(tpc.fl_err1) + std::abs(tpc.fl_err2) +
    // std::abs(tpc.fl_err3) + std::abs(tsc.fl_err));
    ap[0] = std::abs(tpc.fl_err1);
    ap[1] = std::abs(tpc.fl_err2);
    ap[2] = std::abs(tpc.fl_err3);
    ap[3] = std::abs(tsc.fl_err);
    *eb = *eb * std::abs(x) + priest_sum(ap);
  }
  // add error back into result
  *h += e;
  *hd += ed;
}
/* Reversal Horner's Method with Complex Compensated Arithmetic */
void rhorner_comp_cmplx(const std::complex<double> *poly, const std::complex<double> x,
                        const unsigned int deg, std::complex<double> *h, std::complex<double> *hd,
                        double *eb) {
  // local variables
  struct eft eft_arr[6];
  struct eft_cmplx_sum tsc;
  struct eft_cmplx_prod tpc;
  std::complex<double> e = 0, ed = 0;
  std::complex<double> p[4];
  double ap[4];
  // Horner's method
  *h = poly[0];
  *hd = 0;
  *eb = 0;
  for (int i = 1; i <= deg; ++i) {
    // product and sum for derivative evaluation
    two_prod_cmplx(*hd, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, *h, eft_arr, &tsc);
    // update hd and ed
    *hd = tsc.fl_res;
    // ed = ed*x + e + (tpc.fl_err1 + tpc.fl_err2 + tpc.fl_err3 + tsc.fl_err);
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    ed = ed * x + e + priest_cmplx_sum(p);
    // product and sum for polynomial evaluation
    two_prod_cmplx(*h, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, poly[i], eft_arr, &tsc);
    // update h and e
    *h = tsc.fl_res;
    // e = e*x + (tpc.fl_err1 + tpc.fl_err2 + tpc.fl_err3 + tsc.fl_err);
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    e = e * x + priest_cmplx_sum(p);
    // update error bound
    // *eb = *eb*std::abs(x) + (std::abs(tpc.fl_err1) + std::abs(tpc.fl_err2) +
    // std::abs(tpc.fl_err3) + std::abs(tsc.fl_err));
    ap[0] = std::abs(tpc.fl_err1);
    ap[1] = std::abs(tpc.fl_err2);
    ap[2] = std::abs(tpc.fl_err3);
    ap[3] = std::abs(tsc.fl_err);
    *eb = *eb * std::abs(x) + priest_sum(ap);
  }
  // add error back into result
  *h += e;
  *hd += ed;
}
/* Priest CompHorner */
void priest_comp_horner(const std::complex<double> *poly, const std::complex<double> x,
                        const unsigned int deg, std::complex<double> *h) {
  // local variables
  struct eft eft_arr[6];
  struct eft_cmplx_sum tsc;
  struct eft_cmplx_prod tpc;
  std::complex<double> e = 0;
  std::complex<double> p[4];
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    // product and sum for polynomial evaluation
    two_prod_cmplx(*h, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, poly[i], eft_arr, &tsc);
    // update h and e
    *h = tsc.fl_res;
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    e = e * x + priest_cmplx_sum(p);
  }
  // add error back into result
  *h += e;
}
/* AccSum CompHorner */
void accsum_comp_horner(const std::complex<double> *poly, const std::complex<double> x,
                        const unsigned int deg, std::complex<double> *h) {
  // local variables
  struct eft eft_arr[6];
  struct eft_cmplx_sum tsc;
  struct eft_cmplx_prod tpc;
  std::complex<double> e = 0;
  std::complex<double> p[4];
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    // product and sum for polynomial evaluation
    two_prod_cmplx(*h, x, eft_arr, &tpc);
    two_sum_cmplx(tpc.fl_res, poly[i], eft_arr, &tsc);
    // update h and e
    *h = tsc.fl_res;
    p[0] = tpc.fl_err1;
    p[1] = tpc.fl_err2;
    p[2] = tpc.fl_err3;
    p[3] = tsc.fl_err;
    e = e * x + fast_cmplx_acc_sum(p);
  }
  // add error back into result
  *h += e;
}
}  // namespace ehrlich_aberth_jax
#endif
