/* Boost interval/checking.hpp template implementation file
 *
 * Copyright 2002 Hervé Brönnimann, Guillaume Melquiond, Sylvain Pion
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_INTERVAL_CHECKING_HPP
#define BOOST_NUMERIC_INTERVAL_CHECKING_HPP

#include <stdexcept>
#include <string>
#include <cassert>
#include <boost/limits.hpp>

namespace boost {
namespace numeric {
namespace interval_lib {

struct exception_create_empty
{
  void operator()()
  {
    throw std::runtime_error("boost::interval: empty interval created");
  }
};

struct exception_invalid_number
{
  void operator()()
  {
    throw std::invalid_argument("boost::interval: invalid number");
  }
};

template<class T>
struct checking_base
{
  BOOST_GPU_ENABLED static T pos_inf()
  {
    assert(BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_infinity);
    return BOOST_GPU_EQUIV(std::numeric_limits)<T>::infinity();
  }
  BOOST_GPU_ENABLED static T neg_inf()
  {
    assert(BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_infinity);
    return -BOOST_GPU_EQUIV(std::numeric_limits)<T>::infinity();
  }
  BOOST_GPU_ENABLED static T nan()
  {
    assert(BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_quiet_NaN);
    return BOOST_GPU_EQUIV(std::numeric_limits)<T>::quiet_NaN();
  }
  BOOST_GPU_ENABLED static bool is_nan(const T& x)
  {
    return BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_quiet_NaN && (x != x);
  }
  BOOST_GPU_ENABLED static T empty_lower()
  {
    return (BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_quiet_NaN ?
            BOOST_GPU_EQUIV(std::numeric_limits)<T>::quiet_NaN() : static_cast<T>(1));
  }
  BOOST_GPU_ENABLED static T empty_upper()
  {
    return (BOOST_GPU_EQUIV(std::numeric_limits)<T>::has_quiet_NaN ?
            BOOST_GPU_EQUIV(std::numeric_limits)<T>::quiet_NaN() : static_cast<T>(0));
  }
  BOOST_GPU_ENABLED static bool is_empty(const T& l, const T& u)
  {
    return !(l <= u); // safety for partial orders
  }
};

template<class T, class Checking = checking_base<T>,
         class Exception = exception_create_empty>
struct checking_no_empty: Checking
{
  BOOST_GPU_ENABLED static T nan()
  {
    assert(false);
    return Checking::nan();
  }
  BOOST_GPU_ENABLED static T empty_lower()
  {
    #ifndef BOOST_NUMERIC_INTERVAL_USE_GPU
    Exception()();
    return Checking::empty_lower();
    #else
    return checking_base_gpu<T>::empty_lower();
    #endif
  }
  BOOST_GPU_ENABLED static T empty_upper()
  {
    #ifndef BOOST_NUMERIC_INTERVAL_USE_GPU
    Exception()();
    return Checking::empty_upper();
    #else
    return checking_base_gpu<T>::empty_upper();
    #endif
  }
  BOOST_GPU_ENABLED static bool is_empty(const T&, const T&)
  {
    return false;
  }
};

template<class T, class Checking = checking_base<T> >
struct checking_no_nan: Checking
{
  BOOST_GPU_ENABLED static bool is_nan(const T&)
  {
    return false;
  }
};

template<class T, class Checking = checking_base<T>,
         class Exception = exception_invalid_number>
struct checking_catch_nan: Checking
{
  BOOST_GPU_ENABLED static bool is_nan(const T& x)
  {
    if (Checking::is_nan(x)) Exception()();
    return false;
  }
};

template<class T>
struct checking_strict:
  checking_no_nan<T, checking_no_empty<T> >
{};

} // namespace interval_lib
} // namespace numeric
} // namespace boost

#endif // BOOST_NUMERIC_INTERVAL_CHECKING_HPP
