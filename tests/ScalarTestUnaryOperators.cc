/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_unary_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = -a;
    ASSERT_EQ(f.val, -a.val);
    ASSERT_EQ(f.grad, -a.grad);
    if constexpr (with_hessian)
    {
        ASSERT_EQ(f.Hess, -a.Hess);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTestUnaryOperators, UnaryMinusFloatFirstOrder) { test_unary_minus<float, false>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusDoubleFirstOrder) { test_unary_minus<double, false>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusLongDoubleFirstOrder) { test_unary_minus<long double, false>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusFloatSecondOrder) { test_unary_minus<float, true>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusDoubleSecondOrder) { test_unary_minus<double, true>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusLongDoubleSecondOrder) { test_unary_minus<long double, true>(); }

TEST(ScalarTestUnaryOperators, UnaryMinusDoubleFirstOrderDynamic) { test_unary_minus<double, false, true>(); }
TEST(ScalarTestUnaryOperators, UnaryMinusDoubleSecondOrderDynamic) { test_unary_minus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sqrt()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = sqrt(a);
    ASSERT_NEAR(f.val, 2.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 7.0 / 32.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTestUnaryOperators, SqrtFloatFirstOrder) { test_sqrt<float, false>(); }
TEST(ScalarTestUnaryOperators, SqrtDoubleFirstOrder) { test_sqrt<double, false>(); }
TEST(ScalarTestUnaryOperators, SqrtLongDoubleFirstOrder) { test_sqrt<long double, false>(); }
TEST(ScalarTestUnaryOperators, SqrtFloatSecondOrder) { test_sqrt<float, true>(); }
TEST(ScalarTestUnaryOperators, SqrtDoubleSecondOrder) { test_sqrt<double, true>(); }
TEST(ScalarTestUnaryOperators, SqrtLongDoubleSecondOrder) { test_sqrt<long double, true>(); }

TEST(ScalarTestUnaryOperators, SqrtDoubleFirstOrderDynamic) { test_sqrt<double, false, true>(); }
TEST(ScalarTestUnaryOperators, SqrtDoubleSecondOrderDynamic) { test_sqrt<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sqr()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble x = ADouble::make_active(4.0, 0, 2);
    ADouble y = ADouble::make_active(6.0, 1, 2);
    ADouble a = x * x + 7.0 * y * y -3.0 * x * 3.0 + x + 2 * y;

    ADouble a_sqr = sqr(a);
    ADouble a_pow = pow(a, 2);
    ADouble aa = a * a;

    ASSERT_NEAR(a_sqr.val, a_pow.val, 1e-12);
    ASSERT_NEAR(a_sqr.val, aa.val, 1e-12);
    ASSERT_NEAR(a_sqr.grad(0), a_pow.grad(0), 1e-12);
    ASSERT_NEAR(a_sqr.grad(1), a_pow.grad(1), 1e-12);
    ASSERT_NEAR(a_sqr.grad(0), aa.grad(0), 1e-12);
    ASSERT_NEAR(a_sqr.grad(1), aa.grad(1), 1e-12);

    if constexpr (with_hessian)
    {
        TINYAD_ASSERT_SYMMETRIC(a_sqr.Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(a_pow.Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(aa.Hess, 1e-12);
        ASSERT_NEAR(a_sqr.Hess(0, 0), a_pow.Hess(0, 0), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(0, 1), a_pow.Hess(0, 1), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(1, 0), a_pow.Hess(1, 0), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(1, 1), a_pow.Hess(1, 1), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(0, 0), aa.Hess(0, 0), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(0, 1), aa.Hess(0, 1), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(1, 0), aa.Hess(1, 0), 1e-12);
        ASSERT_NEAR(a_sqr.Hess(1, 1), aa.Hess(1, 1), 1e-12);
    }
}

TEST(ScalarTestUnaryOperators, SqrFloatFirstOrder) { test_sqr<float, false>(); }
TEST(ScalarTestUnaryOperators, SqrDoubleFirstOrder) { test_sqr<double, false>(); }
TEST(ScalarTestUnaryOperators, SqrLongDoubleFirstOrder) { test_sqr<long double, false>(); }
TEST(ScalarTestUnaryOperators, SqrFloatSecondOrder) { test_sqr<float, true>(); }
TEST(ScalarTestUnaryOperators, SqrDoubleSecondOrder) { test_sqr<double, true>(); }
TEST(ScalarTestUnaryOperators, SqrLongDoubleSecondOrder) { test_sqr<long double, true>(); }

TEST(ScalarTestUnaryOperators, SqrDoubleFirstOrderDynamic) { test_sqr<double, false, true>(); }
TEST(ScalarTestUnaryOperators, SqrDoubleSecondOrderDynamic) { test_sqr<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_fabs()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;

    {   // a(x) = x^3 at x = 1
        const ADouble a = ADouble::known_derivatives(1.0, 3.0, 6.0);
        const ADouble f = fabs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        const ADouble a = ADouble::known_derivatives(-1.0, 3.0, -6.0);
        const ADouble f = fabs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), -3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }
}

TEST(ScalarTestUnaryOperators, FabsFloatFirstOrder) { test_fabs<float, false>(); }
TEST(ScalarTestUnaryOperators, FabsDoubleFirstOrder) { test_fabs<double, false>(); }
TEST(ScalarTestUnaryOperators, FabsLongDoubleFirstOrder) { test_fabs<long double, false>(); }
TEST(ScalarTestUnaryOperators, FabsFloatSecondOrder) { test_fabs<float, true>(); }
TEST(ScalarTestUnaryOperators, FabsDoubleSecondOrder) { test_fabs<double, true>(); }
TEST(ScalarTestUnaryOperators, FabsLongDoubleSecondOrder) { test_fabs<long double, true>(); }

TEST(ScalarTestUnaryOperators, FabsDoubleFirstOrderDynamic) { test_fabs<double, false, true>(); }
TEST(ScalarTestUnaryOperators, FabsDoubleSecondOrderDynamic) { test_fabs<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_abs()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;

    {   // a(x) = x^3 at x = 1
        const ADouble a = ADouble::known_derivatives(1.0, 3.0, 6.0);
        const ADouble f = abs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        const ADouble a = ADouble::known_derivatives(-1.0, 3.0, -6.0);
        const ADouble f = abs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), -3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }
}

TEST(ScalarTestUnaryOperators, AbsFloatFirstOrder) { test_abs<float, false>(); }
TEST(ScalarTestUnaryOperators, AbsDoubleFirstOrder) { test_abs<double, false>(); }
TEST(ScalarTestUnaryOperators, AbsLongDoubleFirstOrder) { test_abs<long double, false>(); }
TEST(ScalarTestUnaryOperators, AbsFloatSecondOrder) { test_abs<float, true>(); }
TEST(ScalarTestUnaryOperators, AbsDoubleSecondOrder) { test_abs<double, true>(); }
TEST(ScalarTestUnaryOperators, AbsLongDoubleSecondOrder) { test_abs<long double, true>(); }

TEST(ScalarTestUnaryOperators, AbsDoubleFirstOrderDynamic) { test_abs<double, false, true>(); }
TEST(ScalarTestUnaryOperators, AbsDoubleSecondOrderDynamic) { test_abs<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_exp(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = exp(a);
    ASSERT_NEAR(f.val, std::exp(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::exp(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 11.0 * std::exp(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ExpFloatFirstOrder) { test_exp<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ExpDoubleFirstOrder) { test_exp<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ExpLongDoubleFirstOrder) { test_exp<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ExpFloatSecondOrder) { test_exp<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ExpDoubleSecondOrder) { test_exp<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ExpLongDoubleSecondOrder) { test_exp<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ExpDoubleFirstOrderDynamic) { test_exp<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ExpDoubleSecondOrderDynamic) { test_exp<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = log(a);
    ASSERT_NEAR(f.val, 2.0 * std::log(2.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, LogFloatFirstOrder) { test_log<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, LogDoubleFirstOrder) { test_log<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, LogLongDoubleFirstOrder) { test_log<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, LogFloatSecondOrder) { test_log<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, LogDoubleSecondOrder) { test_log<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, LogLongDoubleSecondOrder) { test_log<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, LogDoubleFirstOrderDynamic) { test_log<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, LogDoubleSecondOrderDynamic) { test_log<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log2(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = log2(a);
    ASSERT_NEAR(f.val, 2.0, _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0 / std::log(2.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0 / std::log(2.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, Log2FloatFirstOrder) { test_log2<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, Log2DoubleFirstOrder) { test_log2<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, Log2LongDoubleFirstOrder) { test_log2<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, Log2FloatSecondOrder) { test_log2<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, Log2DoubleSecondOrder) { test_log2<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, Log2LongDoubleSecondOrder) { test_log2<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, Log2DoubleFirstOrderDynamic) { test_log2<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, Log2DoubleSecondOrderDynamic) { test_log2<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log10(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = log10(a);
    ASSERT_NEAR(f.val, std::log10(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0 / std::log(10.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0 / std::log(10.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, Log10FloatFirstOrder) { test_log10<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, Log10DoubleFirstOrder) { test_log10<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, Log10LongDoubleFirstOrder) { test_log10<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, Log10FloatSecondOrder) { test_log10<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, Log10DoubleSecondOrder) { test_log10<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, Log10LongDoubleSecondOrder) { test_log10<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, Log10DoubleFirstOrderDynamic) { test_log10<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, Log10DoubleSecondOrderDynamic) { test_log10<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sin(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = sin(a);
    ASSERT_NEAR(f.val, std::sin(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::cos(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * std::cos(4.0) - 9.0 * std::sin(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, SinFloatFirstOrder) { test_sin<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, SinDoubleFirstOrder) { test_sin<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, SinLongDoubleFirstOrder) { test_sin<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, SinFloatSecondOrder) { test_sin<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, SinDoubleSecondOrder) { test_sin<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, SinLongDoubleSecondOrder) { test_sin<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, SinDoubleFirstOrderDynamic) { test_sin<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, SinDoubleSecondOrderDynamic) { test_sin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_cos(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = cos(a);
    ASSERT_NEAR(f.val, std::cos(4.0), _eps);
    ASSERT_NEAR(f.grad(0), -3.0 * std::sin(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -2.0 * std::sin(4.0) - 9.0 * std::cos(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, CosFloatFirstOrder) { test_cos<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, CosDoubleFirstOrder) { test_cos<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, CosLongDoubleFirstOrder) { test_cos<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, CosFloatSecondOrder) { test_cos<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, CosDoubleSecondOrder) { test_cos<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, CosLongDoubleSecondOrder) { test_cos<double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, CosDoubleFirstOrderDynamic) { test_cos<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, CosDoubleSecondOrderDynamic) { test_cos<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_tan(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = tan(a);
    ASSERT_NEAR(f.val, std::tan(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cos(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 4.0 * (1.0 + 9.0 * std::tan(4.0)) / (1.0 + std::cos(8.0)), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, TanFloatFirstOrder) { test_tan<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, TanDoubleFirstOrder) { test_tan<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, TanLongDoubleFirstOrder) { test_tan<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, TanFloatSecondOrder) { test_tan<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, TanDoubleSecondOrder) { test_tan<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, TanLongDoubleSecondOrder) { test_tan<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, TanDoubleFirstOrderDynamic) { test_tan<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, TanDoubleSecondOrderDynamic) { test_tan<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_asin(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(0.5, 3.0, 2.0);
    const ADouble f = asin(a);
    ASSERT_NEAR(f.val, std::asin(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 3.4641, 1e-4);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 9.2376, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ASinFloatFirstOrder) { test_asin<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ASinDoubleFirstOrder) { test_asin<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinLongDoubleFirstOrder) { test_asin<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinFloatSecondOrder) { test_asin<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ASinDoubleSecondOrder) { test_asin<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinLongDoubleSecondOrder) { test_asin<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ASinDoubleFirstOrderDynamic) { test_asin<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinDoubleSecondOrderDynamic) { test_asin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_acos(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(0.5, 3.0, 2.0);
    const ADouble f = acos(a);
    ASSERT_NEAR(f.val, std::acos(0.5), _eps);
    ASSERT_NEAR(f.grad(0), -3.4641, 1e-4);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -9.2376, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ACosFloatFirstOrder) { test_acos<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ACosDoubleFirstOrder) { test_acos<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ACosLongDoubleFirstOrder) { test_acos<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ACosFloatSecondOrder) { test_acos<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ACosDoubleSecondOrder) { test_acos<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ACosLongDoubleSecondOrder) { test_acos<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ACosDoubleFirstOrderDynamic) { test_asin<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ACosDoubleSecondOrderDynamic) { test_asin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atan(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(0.5, 3.0, 2.0);
    const ADouble f = atan(a);
    ASSERT_NEAR(f.val, std::atan(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 2.4, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -4.16, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ATanFloatFirstOrder) { test_atan<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ATanDoubleFirstOrder) { test_atan<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanLongDoubleFirstOrder) { test_atan<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanFloatSecondOrder) { test_atan<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ATanDoubleSecondOrder) { test_atan<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanLongDoubleSecondOrder) { test_atan<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ATanDoubleFirstOrderDynamic) { test_atan<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanDoubleSecondOrderDynamic) { test_atan<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sinh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = sinh(a);
    ASSERT_NEAR(f.val, std::sinh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::cosh(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 9.0 * std::sinh(4.0) + 2.0 * std::cosh(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, SinhFloatFirstOrder) { test_sinh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, SinhDoubleFirstOrder) { test_sinh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, SinhLongDoubleFirstOrder) { test_sinh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, SinhFloatSecondOrder) { test_sinh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, SinhDoubleSecondOrder) { test_sinh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, SinhLongDoubleSecondOrder) { test_sinh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, SinhDoubleFirstOrderDynamic) { test_sinh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, SinhDoubleSecondOrderDynamic) { test_sinh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_cosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = cosh(a);
    ASSERT_NEAR(f.val, std::cosh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::sinh(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * std::sinh(4.0) + 9.0 * std::cosh(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, CoshFloatFirstOrder) { test_cosh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, CoshDoubleFirstOrder) { test_cosh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, CoshLongDoubleFirstOrder) { test_cosh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, CoshFloatSecondOrder) { test_cosh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, CoshDoubleSecondOrder) { test_cosh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, CoshLongDoubleSecondOrder) { test_cosh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, CoshDoubleFirstOrderDynamic) { test_cosh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, CoshDoubleSecondOrderDynamic) { test_cosh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_tanh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = tanh(a);
    ASSERT_NEAR(f.val, std::tanh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cosh(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * (1.0 - 9.0 * std::sinh(4.0) / std::cosh(4.0)) / (sqr(std::cosh(4.0))), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, TanhFloatFirstOrder) { test_tanh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, TanhDoubleFirstOrder) { test_tanh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, TanhLongDoubleFirstOrder) { test_tanh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, TanhFloatSecondOrder) { test_tanh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, TanhDoubleSecondOrder) { test_tanh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, TanhLongDoubleSecondOrder) { test_tanh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, TanhDoubleFirstOrderDynamic) { test_tanh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, TanhDoubleSecondOrderDynamic) { test_tanh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_asinh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(0.5, 3.0, 2.0);
    const ADouble f = asinh(a);
    ASSERT_NEAR(f.val, std::asinh(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 2.68328, 1e-5);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.43108, 1e-5);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ASinhFloatFirstOrder) { test_asinh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ASinhDoubleFirstOrder) { test_asinh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinhLongDoubleFirstOrder) { test_asinh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinhFloatSecondOrder) { test_asinh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ASinhDoubleSecondOrder) { test_asinh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinhLongDoubleSecondOrder) { test_asinh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ASinhDoubleFirstOrderDynamic) { test_asinh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ASinhDoubleSecondOrderDynamic) { test_asinh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_acosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = acosh(a);
    ASSERT_NEAR(f.val, std::acosh(4), _eps);
    ASSERT_NEAR(f.grad(0), std::sqrt(3.0 / 5.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -2.0 / 5.0 / std::sqrt(15), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ACoshFloatFirstOrder) { test_acosh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ACoshDoubleFirstOrder) { test_acosh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ACoshLongDoubleFirstOrder) { test_acosh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ACoshFloatSecondOrder) { test_acosh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ACoshDoubleSecondOrder) { test_acosh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ACoshLongDoubleSecondOrder) { test_acosh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ACoshDoubleFirstOrderDynamic) { test_acosh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ACoshDoubleSecondOrderDynamic) { test_acosh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atanh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(0.5, 3.0, 2.0);
    const ADouble f = atanh(a);
    ASSERT_NEAR(f.val, std::atanh(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 4.0, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 18.6667, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestUnaryOperators, ATanhFloatFirstOrder) { test_atanh<float, false>(1e-4f); }
TEST(ScalarTestUnaryOperators, ATanhDoubleFirstOrder) { test_atanh<double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanhLongDoubleFirstOrder) { test_atanh<long double, false>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanhFloatSecondOrder) { test_atanh<float, true>(1e-4f); }
TEST(ScalarTestUnaryOperators, ATanhDoubleSecondOrder) { test_atanh<double, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanhLongDoubleSecondOrder) { test_atanh<long double, true>(1e-12); }

TEST(ScalarTestUnaryOperators, ATanhDoubleFirstOrderDynamic) { test_atanh<double, false, true>(1e-12); }
TEST(ScalarTestUnaryOperators, ATanhDoubleSecondOrderDynamic) { test_atanh<double, true, true>(1e-12); }
