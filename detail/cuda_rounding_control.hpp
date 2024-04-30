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

#include <cuda.h>
#include <math_constants.h>
#include <cuda/std/climits>
#include <cuda/std/cassert>
#include <cuda/std/utility>

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
      template <class T> __device__ T sqrt(const T &a) {};

      template <> __device__ float min(const float &a, const float &b)
      {
        return fminf(a, b);
      }
      template <> __device__ float max(const float &a, const float &b)
      {
        return fmaxf(a, b);
      }
      template <> __device__ float floor(const float &a)
      {
        return floorf(a);
      }
      template <> __device__ float ceil(const float &a)
      {
        return ceilf(a);
      }
      template <> __device__ float sin(const float &a)
      {
        return sinf(a);
      }
      template <> __device__ float cos(const float &a)
      {
        return cosf(a);
      }
      template <> __device__ float tan(const float &a)
      {
        return tanf(a);
      }
      template <> __device__ float asin(const float &a)
      {
        return asinf(a);
      }
      template <> __device__ float acos(const float &a)
      {
        return acosf(a);
      }
      template <> __device__ float atan(const float &a)
      {
        return atanf(a);
      }
      template <> __device__ float sinh(const float &a)
      {
        return sinhf(a);
      }
      template <> __device__ float cosh(const float &a)
      {
        return coshf(a);
      }
      template <> __device__ float tanh(const float &a)
      {
        return tanhf(a);
      }
      template <> __device__ float sqrt(const float &a)
      {
        return sqrtf(a);
      }

      template <> __device__ double min(const double &a, const double &b)
      {
        return fmin(a, b);
      }
      template <> __device__ double max(const double &a, const double &b)
      {
        return fmax(a, b);
      }
      template <> __device__ double floor(const double &a)
      {
        return floor(a);
      }
      template <> __device__ double ceil(const double &a)
      {
        return ceil(a);
      }
      template <> __device__ double sin(const double &a)
      {
        return sin(a);
      }
      template <> __device__ double cos(const double &a)
      {
        return cos(a);
      }
      template <> __device__ double tan(const double &a)
      {
        return tan(a);
      }
      template <> __device__ double asin(const double &a)
      {
        return asin(a);
      }
      template <> __device__ double acos(const double &a)
      {
        return acos(a);
      }
      template <> __device__ double atan(const double &a)
      {
        return atan(a);
      }
      template <> __device__ double sinh(const double &a)
      {
        return sinh(a);
      }
      template <> __device__ double cosh(const double &a)
      {
        return cosh(a);
      }
      template <> __device__ double tanh(const double &a)
      {
        return tanh(a);
      }
      template <> __device__ double sqrt(const double &a)
      {
        return sqrt(a);
      }
    } // namespace gpu

    // XXX this is very hacky!
    #undef BOOST_USING_STD_MIN
    #define BOOST_USING_STD_MIN() using gpu::min
    #undef BOOST_USING_STD_MAX
    #define BOOST_USING_STD_MAX() using gpu::max
    #undef BOOST_NUMERIC_INTERVAL_using_math
    #define BOOST_NUMERIC_INTERVAL_using_math(a) using gpu::a

  } // namespace numeric
} // namespace boost

#endif /* BOOST_NUMERIC_INTERVAL_DETAIL_CUDA_ROUNDING_CONTROL_HPP */