/* Boost interval/detail/cuda_rounding_control.hpp file
 *
 * Copyright 2024 Lorenz Gillner
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_INTERVAL_DETAIL_CUDA_ROUNDING_CONTROL_HPP
#define BOOST_NUMERIC_INTERVAL_DETAIL_CUDA_ROUNDING_CONTROL_HPP

#if !defined(__NVCC__) && !defined(__CUDACC__)
#error Boost.Numeric.Interval: This header is intended for CUDA GPUs only.
#endif

#ifndef BOOST_NUMERIC_INTERVAL_USE_GPU
#define BOOST_NUMERIC_INTERVAL_USE_GPU
#endif

#include <boost/numeric/interval/rounding.hpp>

#include <math_constants.h>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/utility>

namespace boost
{
  namespace numeric
  {
    namespace gpu
    {
      template <class T> __device__ T min(const T &a, const T &b) {};
      template <class T> __device__ T max(const T &a, const T &b) {};
      template <class T> __device__ T floor(const T &a) {};
      template <class T> __device__ T ceil(const T &a) {};
      template <class T> __device__ T exp(const T &a) {};
      template <class T> __device__ T log(const T &a) {};
      template <class T> __device__ T sin(const T &a) {};
      template <class T> __device__ T cos(const T &a) {};
      template <class T> __device__ T tan(const T &a) {};
      template <class T> __device__ T asin(const T &a) {};
      template <class T> __device__ T acos(const T &a) {};
      template <class T> __device__ T atan(const T &a) {};
      template <class T> __device__ T sinh(const T &a) {};
      template <class T> __device__ T cosh(const T &a) {};
      template <class T> __device__ T tanh(const T &a) {};
      template <class T> __device__ T asinh(const T &a) {};
      template <class T> __device__ T acosh(const T &a) {};
      template <class T> __device__ T atanh(const T &a) {};
      template <class T> __device__ T sqrt(const T &a) {};

      // XXX isn't there a more elegant way to handle this?
      template <> __device__ float min(const float &a, const float &b) { return fminf(a, b); }
      template <> __device__ float max(const float &a, const float &b) { return fmaxf(a, b); }
      #define BOOST_GPU_SPECIALIZATION(g) template <> __device__ float g(const float &a) { return g##f(a); }
      BOOST_GPU_SPECIALIZATION(floor)
      BOOST_GPU_SPECIALIZATION(ceil)
      BOOST_GPU_SPECIALIZATION(sin)
      BOOST_GPU_SPECIALIZATION(cos)
      BOOST_GPU_SPECIALIZATION(tan)
      BOOST_GPU_SPECIALIZATION(asin)
      BOOST_GPU_SPECIALIZATION(acos)
      BOOST_GPU_SPECIALIZATION(atan)
      BOOST_GPU_SPECIALIZATION(sinh)
      BOOST_GPU_SPECIALIZATION(cosh)
      BOOST_GPU_SPECIALIZATION(tanh)
      BOOST_GPU_SPECIALIZATION(asinh)
      BOOST_GPU_SPECIALIZATION(acosh)
      BOOST_GPU_SPECIALIZATION(atanh)
      BOOST_GPU_SPECIALIZATION(sqrt)
      #undef BOOST_GPU_SPECIALIZATION

      template <> __device__ double min(const double &a, const double &b) { return fmin(a, b); }
      template <> __device__ double max(const double &a, const double &b) { return fmax(a, b); }
      /* sin, cos, tan, etc. have already been defined in CUDA */

    } // namespace gpu
    namespace interval_lib
    {
      /** Checking */

      template<class T>
      struct checking_base_gpu
      {};

      template<>
      struct checking_base_gpu<float>
      {
        static __device__ float pos_inf() { return CUDART_INF_F; }
        static __device__ float neg_inf() { return -CUDART_INF_F; }
        static __device__ float nan() { return CUDART_NAN_F; }
        static __device__ bool is_nan(const float& x) { return (x != x); }
        static __device__ float empty_lower() { return nan(); }
        static __device__ float empty_upper() { return nan(); }
        static __device__ bool is_empty(const float& l, const float& u) { return !(l <= u); }
      };

      template<>
      struct checking_base_gpu<double>
      {
        static __device__ double pos_inf() { return CUDART_INF; }
        static __device__ double neg_inf() { return -CUDART_INF; }
        static __device__ double nan() { return CUDART_NAN; }
        static __device__ bool is_nan(const double& x) { return (x != x); }
        static __device__ double empty_lower() { return nan(); }
        static __device__ double empty_upper() { return nan(); }
        static __device__ bool is_empty(const double& l, const double& u) { return !(l <= u); }
      };

      /** Rounding control */

      // namespace detail
      // {
      //   // arithmetic functions in CUDA are stateless, so this shouldn't do anything
      //   template <class T>
      //   struct cuda_rounding_control
      //   {
      //     typedef unsigned int rounding_mode;
      //     __device__ void set_rounding_mode(rounding_mode &) {}
      //     __device__ void get_rounding_mode(rounding_mode) {}
      //     __device__ void downward() {}
      //     __device__ void upward() {}
      //     __device__ void to_nearest() {}
      //     __device__ void toward_zero() {}
      //     __device__ T force_rounding(const T& r) { return r; }
      //   };

      // } // namespace detail

      // template <class T>
      // struct rounding_control_gpu
      // {
      //   typedef unsigned int rounding_mode;
      //   __device__ void set_rounding_mode(rounding_mode &) {}
      //   __device__ void get_rounding_mode(rounding_mode) {}
      //   __device__ void downward() {}
      //   __device__ void upward() {}
      //   __device__ void to_nearest() {}
      //   __device__ void toward_zero() {}
      //   __device__ const T& to_int(const T& x)         { return x; }
      //   __device__ const T& force_rounding(const T& r) { return r; }
      // };

      template <class T>
      struct rounding_control_gpu: rounding_control<T>
      {};

      template <>
      struct rounding_control_gpu<float>
      {
        __device__ float to_int(const float& x) { return nearbyintf(x); }
      };

      template <>
      struct rounding_control_gpu<double>
      {
        __device__ double to_int(const double& x) { return nearbyint(x); }
      };

      /** Rounded arithmetic functions */

      template <class T, class Rounding = rounding_control_gpu<T> >
      struct rounded_arith_gpu : Rounding
      {
        __device__ void init();
        template <class U>
        __device__ T conv_down(U const &v);
        template <class U>
        __device__ T conv_up(U const &v);
        __device__ T add_down(const T &x, const T &y);
        __device__ T sub_down(const T &x, const T &y);
        __device__ T mul_down(const T &x, const T &y);
        __device__ T div_down(const T &x, const T &y);
        __device__ T add_up(const T &x, const T &y);
        __device__ T sub_up(const T &x, const T &y);
        __device__ T mul_up(const T &x, const T &y);
        __device__ T div_up(const T &x, const T &y);
        __device__ T median(const T &x, const T &y);
        __device__ T sqrt_down(const T &x);
        __device__ T sqrt_up(const T &x);
        __device__ T int_down(const T &x);
        __device__ T int_up(const T &x);
        __device__ T next_down(const T &x);
        __device__ T next_up(const T &x);
      };

      template <class Rounding>
      struct rounded_arith_gpu<float, Rounding>
      {
        __device__ void init() {}
        template <class U>
        __device__ float conv_down(U const &v) { return v; }
        template <class U>
        __device__ float conv_up(U const &v) { return v; }
        __device__ float add_down(const float &x, const float &y) { return __fadd_rd(x, y); }
        __device__ float sub_down(const float &x, const float &y) { return __fsub_rd(x, y); }
        __device__ float mul_down(const float &x, const float &y) { return __fmul_rd(x, y); }
        __device__ float div_down(const float &x, const float &y) { return __fdiv_rd(x, y); }
        __device__ float add_up(const float &x, const float &y) { return __fadd_ru(x, y); }
        __device__ float sub_up(const float &x, const float &y) { return __fsub_ru(x, y); }
        __device__ float mul_up(const float &x, const float &y) { return __fmul_ru(x, y); }
        __device__ float div_up(const float &x, const float &y) { return __fdiv_ru(x, y); }
        __device__ float median(const float &x, const float &y) { return __fdiv_rn(__fadd_rn(x, y), 2); }
        __device__ float sqrt_down(const float &x) { return __fsqrt_rd(x); }
        __device__ float sqrt_up(const float &x) { return __fsqrt_ru(x); }
        __device__ float int_down(const float &x) { return floorf(x); }
        __device__ float int_up(const float &x) { return ceilf(x); }
        __device__ float next_down(const float &x) { return nextafterf(x, -CUDART_INF_F); } // TODO replace with neg_inf()
        __device__ float next_up(const float &x) { return nextafterf(x, CUDART_INF_F); } // TODO replace with pos_inf()
      };

      template <class Rounding>
      struct rounded_arith_gpu<double, Rounding>
      {
        __device__ void init() {}
        template <class U>
        __device__ double conv_down(U const &v) { return v; }
        template <class U>
        __device__ double conv_up(U const &v) { return v; }
        __device__ double add_down(const double &x, const double &y) { return __dadd_rd(x, y); }
        __device__ double sub_down(const double &x, const double &y) { return __dsub_rd(x, y); }
        __device__ double mul_down(const double &x, const double &y) { return __dmul_rd(x, y); }
        __device__ double div_down(const double &x, const double &y) { return __ddiv_rd(x, y); }
        __device__ double add_up(const double &x, const double &y) { return __dadd_ru(x, y); }
        __device__ double sub_up(const double &x, const double &y) { return __dsub_ru(x, y); }
        __device__ double mul_up(const double &x, const double &y) { return __dmul_ru(x, y); }
        __device__ double div_up(const double &x, const double &y) { return __ddiv_ru(x, y); }
        __device__ double median(const double &x, const double &y) { return __ddiv_rn(__dadd_rn(x, y), 2); }
        __device__ double sqrt_down(const double &x) { return __dsqrt_rd(x); }
        __device__ double sqrt_up(const double &x) { return __dsqrt_ru(x); }
        __device__ double int_down(const double &x) { return floor(x); }
        __device__ double int_up(const double &x) { return ceil(x); }
        __device__ double next_down(const double &x) { return nextafter(x, -CUDART_INF); } // TODO replace with neg_inf()
        __device__ double next_up(const double &x) { return nextafter(x, CUDART_INF); } // TODO replace with pos_inf()
      };

      /** Rounded transcendental functions */

      template <class T, class Rounding>
      struct rounded_transc_gpu: Rounding
      {
        /* # define BOOST_NUMERIC_INTERVAL_new_func(f) \
          __device__ T f##_down(const T& x) { BOOST_NUMERIC_INTERVAL_using_math(f); return next_down(f(x)); } \
          __device__ T f##_up  (const T& x) { BOOST_NUMERIC_INTERVAL_using_math(f); return next_up(f(x)); } */
          # define BOOST_NUMERIC_INTERVAL_new_func(f) \
          __device__ T f##_down(const T& x) { return next_down(boost::numeric::gpu::f(x)); } \
          __device__ T f##_up  (const T& x) { return next_up(boost::numeric::gpu::f(x)); }
        BOOST_NUMERIC_INTERVAL_new_func(exp)
        BOOST_NUMERIC_INTERVAL_new_func(log)
        BOOST_NUMERIC_INTERVAL_new_func(sin)
        BOOST_NUMERIC_INTERVAL_new_func(cos)
        BOOST_NUMERIC_INTERVAL_new_func(tan)
        BOOST_NUMERIC_INTERVAL_new_func(asin)
        BOOST_NUMERIC_INTERVAL_new_func(acos)
        BOOST_NUMERIC_INTERVAL_new_func(atan)
        BOOST_NUMERIC_INTERVAL_new_func(sinh)
        BOOST_NUMERIC_INTERVAL_new_func(cosh)
        BOOST_NUMERIC_INTERVAL_new_func(tanh)
        BOOST_NUMERIC_INTERVAL_new_func(asinh)
        BOOST_NUMERIC_INTERVAL_new_func(acosh)
        BOOST_NUMERIC_INTERVAL_new_func(atanh)
        #undef BOOST_NUMERIC_INTERVAL_new_func
      };

      template <class T>
      struct rounded_math_gpu: save_state_nothing<rounded_arith_exact<T> >
      {};

      template <>
      struct rounded_math_gpu<float>: save_state_nothing<rounded_arith_gpu<float> >
      {};

      template <>
      struct rounded_math_gpu<double>: save_state_nothing<rounded_arith_gpu<double> >
      {};
      
      template<class T, class Rounding = rounded_arith_gpu<T> > 
      struct rounded_transc_gpu;

      template<class T>
      struct default_policies_gpu
      {
        typedef policies<rounded_math_gpu<T>, checking_base_gpu<T> > type;
      };
    } // namespace interval_lib

    #if defined(__CUDA_ARCH__)
    #  undef BOOST_USING_STD_MIN
    #  define BOOST_USING_STD_MIN() using gpu::min
    #  undef BOOST_USING_STD_MAX
    #  define BOOST_USING_STD_MAX() using gpu::max
    #endif
    // #undef BOOST_USING_STD_MIN
    // #define BOOST_USING_STD_MIN() using cuda::std::min;
    // #undef BOOST_USING_STD_MAX
    // #define BOOST_USING_STD_MAX() using cuda::std::max;

    #undef BOOST_NUMERIC_INTERVAL_using_math
    #define BOOST_NUMERIC_INTERVAL_using_math(a) using gpu::a

  } // namespace numeric
} // namespace boost

#endif /* BOOST_NUMERIC_INTERVAL_DETAIL_CUDA_ROUNDING_CONTROL_HPP */