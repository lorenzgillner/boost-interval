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

#include <cuda.h>
#include <math_constants.h>
#include <cuda/std/climits>
#include <cuda/std/cassert>

namespace boost
{
  namespace numeric
  {
    namespace interval_lib
    {
      /** GPU checking */
      template<class T>
      struct checking_base_gpu
      {};

      template<>
      struct checking_base_gpu<float>
      {
        __device__ static float pos_inf()
        {
          return CUDART_INF_F;
        }
        __device__ static float neg_inf()
        {
          return -CUDART_INF_F;
        }
        __device__ static float nan()
        {
          return CUDART_NAN_F;
        }
        __device__ static bool is_nan(const float& x)
        {
          return (x != x);
        }
        __device__ static float empty_lower()
        {
          return nan();
        }
        __device__ static float empty_upper()
        {
          return nan();
        }
        __device__ static bool is_empty(const float& l, const float& u)
        {
          return !(l <= u);
        }
      };

      template<>
      struct checking_base_gpu<double>
      {
        __device__ static double pos_inf()
        {
          return CUDART_INF;
        }
        __device__ static double neg_inf()
        {
          return -CUDART_INF;
        }
        __device__ static double nan()
        {
          return CUDART_NAN;
        }
        __device__ static bool is_nan(const double& x)
        {
          return (x != x);
        }
        __device__ static double empty_lower()
        {
          return nan();
        }
        __device__ static double empty_upper()
        {
          return nan();
        }
        __device__ static bool is_empty(const double& l, const double& u)
        {
          return !(l <= u);
        }
      };

      /** GPU rounding control */
      namespace detail
      {
        // arithmetic functions in CUDA are stateless, so this shouldn't do anything
        struct cuda_rounding_control
        {
          typedef unsigned int rounding_mode;
          __device__ void set_rounding_mode(rounding_mode &) {}
          __device__ void get_rounding_mode(rounding_mode) {}
          __device__ void downward() {}
          __device__ void upward() {}
          __device__ void to_nearest() {}
          __device__ void toward_zero() {}
        };

      } // namespace detail

      template <class T>
      struct rounding_control_gpu: detail::cuda_rounding_control
      {};

      template <>
      struct rounding_control_gpu<float>: detail::cuda_rounding_control
      {
        __device__ float to_int(const float& x)
        {
          return nearbyintf(x);
        }
        __device__ float force_rounding(const float& r)
        {
          return r;
        }
      };

      template <>
      struct rounding_control_gpu<double>: detail::cuda_rounding_control
      {
        __device__ double to_int(const double& x)
        {
          return nearbyint(x);
        }
        __device__ double force_rounding(const double& r)
        {
          return r;
        }
      };

      // template<class T, class Rounding = rounding_control_gpu<T> >
      // struct rounded_arith_gpu;

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
      };

      template <class Rounding>
      struct rounded_arith_gpu<float, Rounding>
      {
#define BOOST_DN(EXPR) __fadd_rd(EXPR, 0) // is this legal?
#define BOOST_UP(EXPR) __fadd_ru(EXPR, 0)
        __device__ void init() {}
        template <class U>
        __device__ float conv_down(U const &v) { return BOOST_DN(v); }
        template <class U>
        __device__ float conv_up(U const &v) { return BOOST_UP(v); }
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
        __device__ float int_down(const float &x) { return nearbyintf(BOOST_DN(x)); }
        __device__ float int_up(const float &x) { return nearbyintf(BOOST_UP(x)); }
#undef BOOST_DN
#undef BOOST_UP
      };

      template <class Rounding>
      struct rounded_arith_gpu<double, Rounding>
      {
#define BOOST_DN(EXPR) __dadd_rd(EXPR, 0)
#define BOOST_UP(EXPR) __dadd_ru(EXPR, 0)
        __device__ void init() {}
        template <class U>
        __device__ double conv_down(U const &v) { return BOOST_DN(v); }
        template <class U>
        __device__ double conv_up(U const &v) { return BOOST_UP(v); }
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
        __device__ double int_down(const double &x) { return nearbyint(BOOST_DN(x)); }
        __device__ double int_up(const double &x) { return nearbyint(BOOST_UP(x)); }
#undef BOOST_DN
#undef BOOST_UP
      };

      // TODO rounded_transc

      template <class T>
      struct rounded_math_gpu: save_state_nothing<rounded_arith_exact<T> >
      {};

      template <>
      struct rounded_math_gpu<float>: save_state_nothing<rounded_arith_gpu<float> >
      {};

      template <>
      struct rounded_math_gpu<double>: save_state_nothing<rounded_arith_gpu<double> >
      {};

      template<class T>
      struct default_policies_gpu
      {
        typedef policies<rounded_math_gpu<T>, checking_base_gpu<T> > type;
      };
    } // namespace interval_lib
  } // namespace numeric
} // namespace boost

#endif /* BOOST_NUMERIC_INTERVAL_DETAIL_CUDA_ROUNDING_CONTROL_HPP */