/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <sstream>
#include <iostream>

namespace TinyAD
{

// ///////////////////////////////////////////////////////////////////////////
// Debugging options
// ///////////////////////////////////////////////////////////////////////////

#define TINYAD_ENABLE_OPERATOR_LOGGING 0
#define TINYAD_ENABLE_FINITE_CHECKS 0

// ///////////////////////////////////////////////////////////////////////////
// Assertions and debug macros
// ///////////////////////////////////////////////////////////////////////////

#define TINYAD_ANSI_FG_MAGENTA "\x1b[35m"
#define TINYAD_ANSI_FG_YELLOW "\x1b[33m"
#define TINYAD_ANSI_FG_GREEN "\x1b[32m"
#define TINYAD_ANSI_FG_WHITE "\x1b[37m"
#define TINYAD_ANSI_FG_RED "\x1b[31m"
#define TINYAD_ANSI_RESET "\x1b[0m"


#define TINYAD_INFO(str) \
{ \
    std::cout << TINYAD_ANSI_FG_GREEN << str << TINYAD_ANSI_RESET << std::endl; \
    std::cout.flush(); \
}

#define TINYAD_DEBUG_OUT(str) \
{ \
    std::cout << TINYAD_ANSI_FG_MAGENTA \
              << "[DEBUG] " \
              << str \
              << TINYAD_ANSI_RESET << std::endl; \
    std::cout.flush(); \
}

#define TINYAD_DEBUG_VAR(var) \
{ \
    TINYAD_DEBUG_OUT(#var << " = " << var) \
}

#define TINYAD_WARNING(str) \
{ \
    std::cout << TINYAD_ANSI_FG_YELLOW \
              << "[WARNING] " \
              << str \
              << TINYAD_ANSI_RESET \
              << " (in function " << __FUNCTION__ << ":" << __LINE__ \
              << " in file " << __FILE__ << ")" \
              << std::endl; \
    std::cout.flush(); \
}

#define TINYAD_ERROR(str) \
    std::cout << TINYAD_ANSI_FG_RED \
              << "[ERROR] " \
              << str \
              << TINYAD_ANSI_RESET \
              << " (in function " << __FUNCTION__ << ":" << __LINE__ \
              << " in file " << __FILE__ << ")" \
              << std::endl

#define TINYAD_ERROR_throw(st) \
{ \
    TINYAD_ERROR(st); \
    std::stringstream str_strm; \
    str_strm << "[ERROR] " << st; \
    throw std::runtime_error(str_strm.str()); \
}

#define TINYAD_ASSERT(exp) \
{ \
    if(!(exp)) TINYAD_ERROR_throw("Assertion failed: " << (#exp)); \
}

#define TINYAD_ASSERT_EQ(a, b) \
{ \
    if((a) != (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " == " << (b)); \
}

#define TINYAD_ASSERT_NEQ(a, b) \
{ \
    if((a) == (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " != " << (b)); \
}

#define TINYAD_ASSERT_G(a, b) \
{ \
    if((a) <= (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " > " << (b)); \
}

#define TINYAD_ASSERT_GEQ(a, b) \
{ \
    if((a) < (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " >= " << (b)); \
}

#define TINYAD_ASSERT_L(a, b) \
{ \
    if((a) >= (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " < " << (b)); \
}

#define TINYAD_ASSERT_LEQ(a, b) \
{ \
    if((a) > (b)) TINYAD_ERROR_throw("Assertion failed: " << (a) << " <= " << (b)); \
}

#define TINYAD_ASSERT_EPS(a, b, eps) \
{ \
    if(std::abs((a) - (b)) >= eps) TINYAD_ERROR_throw("Assertion failed: |" << (a) << " - " << (b) << "| < " << eps); \
}

#define TINYAD_ASSERT_EPS_MAT(A, B, eps) \
{ \
    const auto& A_ref = A; \
    const auto& B_ref = B; \
    TINYAD_ASSERT_EQ(A_ref.rows(), B_ref.rows()); \
    TINYAD_ASSERT_EQ(A_ref.cols(), B_ref.cols()); \
    for (Eigen::Index i = 0; i < A_ref.rows(); ++i) \
    { \
        for (Eigen::Index j = 0; j < A_ref.cols(); ++j) \
            TINYAD_ASSERT_EPS(A_ref(i, j), B_ref(i, j), eps); \
    } \
}

#define TINYAD_ASSERT_FINITE(a) \
{ \
    TINYAD_ASSERT(std::isfinite(a)); \
}

#define TINYAD_ASSERT_FINITE_MAT(A) \
{ \
    const auto& A_ref = A; \
    for (Eigen::Index i = 0; i < A_ref.rows(); ++i) \
    { \
        for (Eigen::Index j = 0; j < A_ref.cols(); ++j) \
        { \
            if (!std::isfinite(A_ref(i, j))) \
                TINYAD_ERROR_throw("Assertion failed: Not finite " << A_ref); \
        } \
    } \
}

#define TINYAD_ASSERT_SYMMETRIC(A, eps) \
{ \
    const auto& A_ref = A; \
    if (((A_ref) - (A_ref).transpose()).array().abs().maxCoeff() > eps) \
        TINYAD_ERROR_throw("Matrix not symmetric"); \
}

/// NAN-check for double type
#if (TINYAD_ENABLE_FINITE_CHECKS)
#define TINYAD_CHECK_FINITE_IF_ENABLED_d(exp) TINYAD_ASSERT_FINITE(exp);
#else
#define TINYAD_CHECK_FINITE_IF_ENABLED_d(exp) { }
#endif

/// NAN-check for TinyAD::Scalar type
#if (TINYAD_ENABLE_FINITE_CHECKS)
#define TINYAD_CHECK_FINITE_IF_ENABLED_AD(exp) \
{ \
    const auto& exp_ref = exp; \
    TINYAD_ASSERT_FINITE(exp_ref.val); \
    TINYAD_ASSERT_FINITE_MAT(exp_ref.grad); \
    TINYAD_ASSERT_FINITE_MAT(exp_ref.Hess); \
}
#else
#define TINYAD_CHECK_FINITE_IF_ENABLED_AD(exp) { }
#endif

}
