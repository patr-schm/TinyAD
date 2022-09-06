/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool with_hessian>
void test_constructors()
{
    static_assert(std::is_copy_constructible<TinyAD::Scalar<6, PassiveT>>::value, "");
    static_assert(std::is_move_constructible<TinyAD::Scalar<6, PassiveT>>::value, "");
    static_assert(std::is_copy_assignable<TinyAD::Scalar<6, PassiveT>>::value, "");
    static_assert(std::is_move_assignable<TinyAD::Scalar<6, PassiveT>>::value, "");

    {
        // Active variable
        TinyAD::Scalar<2, PassiveT, with_hessian> a(4.0, 0);
        ASSERT_EQ(a.val, 4.0);
        ASSERT_EQ(a.grad[0], 1.0);
        ASSERT_EQ(a.grad[1], 0.0);
        ASSERT_TRUE(a.Hess.isZero());

        // Passive variable
        TinyAD::Scalar<2, PassiveT, with_hessian> b(5.0);
        ASSERT_EQ(b.val, 5.0);
        ASSERT_TRUE(b.grad.isZero());
        ASSERT_TRUE(b.Hess.isZero());

        // Copy constructor
        const auto a2(a);
        ASSERT_EQ(a.val, a2.val);
        ASSERT_EQ(a.grad, a2.grad);
        ASSERT_EQ(a.Hess, a2.Hess);

        // Assignment operator
        const auto b2 = b;
        ASSERT_EQ(b.val, b2.val);
        ASSERT_EQ(b.grad, b2.grad);
        ASSERT_EQ(b.Hess, b2.Hess);
    }
}

TEST(ScalarTest, ConstructorsFloatFirstOrder) { test_constructors<float, false>(); }
TEST(ScalarTest, ConstructorsDoubleFirstOrder) { test_constructors<double, false>(); }
TEST(ScalarTest, ConstructorsLongDoubleFirstOrder) { test_constructors<long double, false>(); }
TEST(ScalarTest, ConstructorsFloatSecondOrder) { test_constructors<float, true>(); }
TEST(ScalarTest, ConstructorsDoubleSecondOrder) { test_constructors<double, true>(); }
TEST(ScalarTest, ConstructorsLongDoubleSecondOrder) { test_constructors<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_to_passive()
{
    {
        // make_active()
        const auto v = TinyAD::Scalar<2, PassiveT, with_hessian>::make_active({ 2.0, 4.0 });
        ASSERT_EQ(v[0].val, 2.0);
        ASSERT_EQ(v[1].val, 4.0);
        ASSERT_EQ(v[0].grad[0], 1.0);
        ASSERT_EQ(v[0].grad[1], 0.0);
        ASSERT_EQ(v[1].grad[0], 0.0);
        ASSERT_EQ(v[1].grad[1], 1.0);
        ASSERT_TRUE(v[0].Hess.isZero());
        ASSERT_TRUE(v[1].Hess.isZero());

        // to_passive() vector
        const Eigen::Matrix<PassiveT, 2, 1> v_passive = TinyAD::to_passive(v);
        const Eigen::Matrix<PassiveT, 2, 1> v_passive2 = TinyAD::to_passive(v_passive);
        TINYAD_ASSERT_EQ(v_passive[0], 2.0);
        TINYAD_ASSERT_EQ(v_passive[1], 4.0);
        TINYAD_ASSERT_EQ(v_passive2[0], 2.0);
        TINYAD_ASSERT_EQ(v_passive2[1], 4.0);
    }

    {
        // to_passive() matrix
        using ActiveT = TinyAD::Scalar<4, PassiveT, with_hessian>;
        const Eigen::Vector<ActiveT, 4> v = ActiveT::make_active({ 1.0, 2.0, 3.0, 4.0 });
        Eigen::Matrix<ActiveT, 2, 2> M;
        M << v[0], v[1], v[2], v[3];
        const Eigen::Matrix2<PassiveT> M_passive = TinyAD::to_passive(M);
        const Eigen::Matrix2<PassiveT> M_passive2 = TinyAD::to_passive(M_passive);
        ASSERT_EQ(M(0, 0).val, M_passive(0, 0));
        ASSERT_EQ(M(0, 1).val, M_passive(0, 1));
        ASSERT_EQ(M(1, 0).val, M_passive(1, 0));
        ASSERT_EQ(M(1, 1).val, M_passive(1, 1));
        ASSERT_EQ(M_passive2(0, 0), M_passive(0, 0));
        ASSERT_EQ(M_passive2(0, 1), M_passive(0, 1));
        ASSERT_EQ(M_passive2(1, 0), M_passive(1, 0));
        ASSERT_EQ(M_passive2(1, 1), M_passive(1, 1));
    }
}

TEST(ScalarTest, PassiveToPassiveFloatFirstOrder) { test_to_passive<float, false>(); }
TEST(ScalarTest, PassiveToPassiveDoubleFirstOrder) { test_to_passive<double, false>(); }
TEST(ScalarTest, PassiveToPassiveLongDoubleFirstOrder) { test_to_passive<long double, false>(); }
TEST(ScalarTest, PassiveToPassiveFloatSecondOrder) { test_to_passive<float, true>(); }
TEST(ScalarTest, PassiveToPassiveDoubleSecondOrder) { test_to_passive<double, true>(); }
TEST(ScalarTest, PassiveToPassiveLongDoubleSecondOrder) { test_to_passive<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_quadratic()
{
    // f(a) = a^2 + a + 2 at a=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(1.0, 0);
    const auto f = sqr(a) + a + 2.0;
    ASSERT_EQ(f.val, 4.0);
    ASSERT_EQ(f.grad(0), 3.0);
    if constexpr (with_hessian)
    {
        ASSERT_EQ(f.Hess(0, 0), 2.0);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTest, QuadraticFloatFirstOrder) { test_quadratic<float, false>();}
TEST(ScalarTest, QuadraticDoubleFirstOrder) { test_quadratic<double, false>();}
TEST(ScalarTest, QuadraticLongDoubleFirstOrder) { test_quadratic<long double, false>();}
TEST(ScalarTest, QuadraticFloatSecondOrder) { test_quadratic<float, true>();}
TEST(ScalarTest, QuadraticDoubleSecondOrder) { test_quadratic<double, true>();}
TEST(ScalarTest, QuadraticLongDoubleSecondOrder) { test_quadratic<long double, true>();}

template <typename PassiveT, bool with_hessian>
void test_unary_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = -a;
    ASSERT_EQ(f.val, -a.val);
    ASSERT_EQ(f.grad, -a.grad);
    if constexpr (with_hessian)
    {
        ASSERT_EQ(f.Hess, -a.Hess);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTest, UnaryMinusFloatFirstOrder) { test_unary_minus<float, false>(); }
TEST(ScalarTest, UnaryMinusDoubleFirstOrder) { test_unary_minus<double, false>(); }
TEST(ScalarTest, UnaryMinusLongDoubleFirstOrder) { test_unary_minus<long double, false>(); }
TEST(ScalarTest, UnaryMinusFloatSecondOrder) { test_unary_minus<float, true>(); }
TEST(ScalarTest, UnaryMinusDoubleSecondOrder) { test_unary_minus<double, true>(); }
TEST(ScalarTest, UnaryMinusLongDoubleSecondOrder) { test_unary_minus<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_sqrt()
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = sqrt(a);
    ASSERT_NEAR(f.val, 2.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 7.0 / 32.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTest, SqrtFloatFirstOrder) { test_sqrt<float, false>(); }
TEST(ScalarTest, SqrtDoubleFirstOrder) { test_sqrt<double, false>(); }
TEST(ScalarTest, SqrtLongDoubleFirstOrder) { test_sqrt<long double, false>(); }
TEST(ScalarTest, SqrtFloatSecondOrder) { test_sqrt<float, true>(); }
TEST(ScalarTest, SqrtDoubleSecondOrder) { test_sqrt<double, true>(); }
TEST(ScalarTest, SqrtLongDoubleSecondOrder) { test_sqrt<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_sqr()
{
    using AD = TinyAD::Scalar<2, PassiveT, with_hessian>;
    AD x(4.0, 0);
    AD y(6.0, 1);
    AD a = x * x + 7.0 * y * y -3.0 * x * 3.0 + x + 2 * y;

    AD a_sqr = sqr(a);
    AD a_pow = pow(a, 2);
    AD aa = a * a;

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

TEST(ScalarTest, SqrFloatFirstOrder) { test_sqr<float, false>(); }
TEST(ScalarTest, SqrDoubleFirstOrder) { test_sqr<double, false>(); }
TEST(ScalarTest, SqrLongDoubleFirstOrder) { test_sqr<long double, false>(); }
TEST(ScalarTest, SqrFloatSecondOrder) { test_sqr<float, true>(); }
TEST(ScalarTest, SqrDoubleSecondOrder) { test_sqr<double, true>(); }
TEST(ScalarTest, SqrLongDoubleSecondOrder) { test_sqr<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_pow_int()
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = pow(a, 3);
    ASSERT_NEAR(f.val, 64.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 144.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 312.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTest, PowFloatIntFirstOrder) { test_pow_int<float, false>(); }
TEST(ScalarTest, PowDoubleIntFirstOrder) { test_pow_int<double, false>(); }
TEST(ScalarTest, PowLongDoubleIntFirstOrder) { test_pow_int<long double, false>(); }
TEST(ScalarTest, PowFloatIntSecondOrder) { test_pow_int<float, true>(); }
TEST(ScalarTest, PowDoubleIntSecondOrder) { test_pow_int<double, true>(); }
TEST(ScalarTest, PowLongDoubleIntSecondOrder) { test_pow_int<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_pow_real()
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = pow(a, PassiveT(3.0 / 2.0));
    ASSERT_NEAR(f.val, 8.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 9.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 75.0 / 8.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTest, PowFloatRealFirstOrder) { test_pow_real<float, false>(); }
TEST(ScalarTest, PowDoubleRealFirstOrder) { test_pow_real<double, false>(); }
TEST(ScalarTest, PowLongDoubleRealFirstOrder) { test_pow_real<long double, false>(); }
TEST(ScalarTest, PowFloatRealSecondOrder) { test_pow_real<float, true>(); }
TEST(ScalarTest, PowDoubleRealSecondOrder) { test_pow_real<double, true>(); }
TEST(ScalarTest, PowLongDoubleRealSecondOrder) { test_pow_real<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_fabs()
{
    {   // a(x) = x^3 at x = 1
        TinyAD::Scalar<1, PassiveT, with_hessian> a(1.0, 3.0, 6.0);
        const auto f = fabs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        TinyAD::Scalar<1, PassiveT, with_hessian> a(-1.0, 3.0, -6.0);
        const auto f = fabs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), -3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, FabsFloatFirstOrder) { test_fabs<float, false>(); }
TEST(ScalarTest, FabsDoubleFirstOrder) { test_fabs<double, false>(); }
TEST(ScalarTest, FabsLongDoubleFirstOrder) { test_fabs<long double, false>(); }
TEST(ScalarTest, FabsFloatSecondOrder) { test_fabs<float, true>(); }
TEST(ScalarTest, FabsDoubleSecondOrder) { test_fabs<double, true>(); }
TEST(ScalarTest, FabsLongDoubleSecondOrder) { test_fabs<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_abs()
{
    {   // a(x) = x^3 at x = 1
        TinyAD::Scalar<1, PassiveT, with_hessian> a(1.0, 3.0, 6.0);
        const auto f = abs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        TinyAD::Scalar<1, PassiveT, with_hessian> a(-1.0, 3.0, -6.0);
        const auto f = abs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), -3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, AbsFloatFirstOrder) { test_abs<float, false>(); }
TEST(ScalarTest, AbsDoubleFirstOrder) { test_abs<double, false>(); }
TEST(ScalarTest, AbsLongDoubleFirstOrder) { test_abs<long double, false>(); }
TEST(ScalarTest, AbsFloatSecondOrder) { test_abs<float, true>(); }
TEST(ScalarTest, AbsDoubleSecondOrder) { test_abs<double, true>(); }
TEST(ScalarTest, AbsLongDoubleSecondOrder) { test_abs<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_exp(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = exp(a);
    ASSERT_NEAR(f.val, std::exp(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::exp(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 11.0 * std::exp(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ExpFloatFirstOrder) { test_exp<float, false>(1e-4f); }
TEST(ScalarTest, ExpDoubleFirstOrder) { test_exp<double, false>(1e-12); }
TEST(ScalarTest, ExpLongDoubleFirstOrder) { test_exp<long double, false>(1e-12); }
TEST(ScalarTest, ExpFloatSecondOrder) { test_exp<float, true>(1e-4f); }
TEST(ScalarTest, ExpDoubleSecondOrder) { test_exp<double, true>(1e-12); }
TEST(ScalarTest, ExpLongDoubleSecondOrder) { test_exp<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_log(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = log(a);
    ASSERT_NEAR(f.val, 2.0 * std::log(2.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, LogFloatFirstOrder) { test_log<float, false>(1e-4f); }
TEST(ScalarTest, LogDoubleFirstOrder) { test_log<double, false>(1e-12); }
TEST(ScalarTest, LogLongDoubleFirstOrder) { test_log<long double, false>(1e-12); }
TEST(ScalarTest, LogFloatSecondOrder) { test_log<float, true>(1e-4f); }
TEST(ScalarTest, LogDoubleSecondOrder) { test_log<double, true>(1e-12); }
TEST(ScalarTest, LogLongDoubleSecondOrder) { test_log<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_log2(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = log2(a);
    ASSERT_NEAR(f.val, 2.0, _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0 / std::log(2.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0 / std::log(2.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, Log2FloatFirstOrder) { test_log2<float, false>(1e-4f); }
TEST(ScalarTest, Log2DoubleFirstOrder) { test_log2<double, false>(1e-12); }
TEST(ScalarTest, Log2LongDoubleFirstOrder) { test_log2<long double, false>(1e-12); }
TEST(ScalarTest, Log2FloatSecondOrder) { test_log2<float, true>(1e-4f); }
TEST(ScalarTest, Log2DoubleSecondOrder) { test_log2<double, true>(1e-12); }
TEST(ScalarTest, Log2LongDoubleSecondOrder) { test_log2<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_log10(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = log10(a);
    ASSERT_NEAR(f.val, std::log10(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / 4.0 / std::log(10.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0 / std::log(10.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, Log10FloatFirstOrder) { test_log10<float, false>(1e-4f); }
TEST(ScalarTest, Log10DoubleFirstOrder) { test_log10<double, false>(1e-12); }
TEST(ScalarTest, Log10LongDoubleFirstOrder) { test_log10<long double, false>(1e-12); }
TEST(ScalarTest, Log10FloatSecondOrder) { test_log10<float, true>(1e-4f); }
TEST(ScalarTest, Log10DoubleSecondOrder) { test_log10<double, true>(1e-12); }
TEST(ScalarTest, Log10LongDoubleSecondOrder) { test_log10<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_sin(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = sin(a);
    ASSERT_NEAR(f.val, std::sin(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::cos(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * std::cos(4.0) - 9.0 * std::sin(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, SinFloatFirstOrder) { test_sin<float, false>(1e-4f); }
TEST(ScalarTest, SinDoubleFirstOrder) { test_sin<double, false>(1e-12); }
TEST(ScalarTest, SinLongDoubleFirstOrder) { test_sin<long double, false>(1e-12); }
TEST(ScalarTest, SinFloatSecondOrder) { test_sin<float, true>(1e-4f); }
TEST(ScalarTest, SinDoubleSecondOrder) { test_sin<double, true>(1e-12); }
TEST(ScalarTest, SinLongDoubleSecondOrder) { test_sin<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_cos(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = cos(a);
    ASSERT_NEAR(f.val, std::cos(4.0), _eps);
    ASSERT_NEAR(f.grad(0), -3.0 * std::sin(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -2.0 * std::sin(4.0) - 9.0 * std::cos(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, CosFloatFirstOrder) { test_cos<float, false>(1e-4f); }
TEST(ScalarTest, CosDoubleFirstOrder) { test_cos<double, false>(1e-12); }
TEST(ScalarTest, CosLongDoubleFirstOrder) { test_cos<double, false>(1e-12); }
TEST(ScalarTest, CosFloatSecondOrder) { test_cos<float, true>(1e-4f); }
TEST(ScalarTest, CosDoubleSecondOrder) { test_cos<double, true>(1e-12); }
TEST(ScalarTest, CosLongDoubleSecondOrder) { test_cos<double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_tan(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = tan(a);
    ASSERT_NEAR(f.val, std::tan(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cos(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 4.0 * (1.0 + 9.0 * std::tan(4.0)) / (1.0 + std::cos(8.0)), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, PassiveTanFloatFirstOrder) { test_tan<float, false>(1e-4f); }
TEST(ScalarTest, PassiveTanDoubleFirstOrder) { test_tan<double, false>(1e-12); }
TEST(ScalarTest, PassiveTanLongDoubleFirstOrder) { test_tan<long double, false>(1e-12); }
TEST(ScalarTest, PassiveTanFloatSecondOrder) { test_tan<float, true>(1e-4f); }
TEST(ScalarTest, PassiveTanDoubleSecondOrder) { test_tan<double, true>(1e-12); }
TEST(ScalarTest, PassiveTanLongDoubleSecondOrder) { test_tan<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_asin(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const auto f = asin(a);
    ASSERT_NEAR(f.val, std::asin(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 3.4641, 1e-4);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 9.2376, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ASinFloatFirstOrder) { test_asin<float, false>(1e-4f); }
TEST(ScalarTest, ASinDoubleFirstOrder) { test_asin<double, false>(1e-12); }
TEST(ScalarTest, ASinLongDoubleFirstOrder) { test_asin<long double, false>(1e-12); }
TEST(ScalarTest, ASinFloatSecondOrder) { test_asin<float, true>(1e-4f); }
TEST(ScalarTest, ASinDoubleSecondOrder) { test_asin<double, true>(1e-12); }
TEST(ScalarTest, ASinLongDoubleSecondOrder) { test_asin<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_acos(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const auto f = acos(a);
    ASSERT_NEAR(f.val, std::acos(0.5), _eps);
    ASSERT_NEAR(f.grad(0), -3.4641, 1e-4);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -9.2376, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ACosFloatFirstOrder) { test_acos<float, false>(1e-4f); }
TEST(ScalarTest, ACosDoubleFirstOrder) { test_acos<double, false>(1e-12); }
TEST(ScalarTest, ACosLongDoubleFirstOrder) { test_acos<long double, false>(1e-12); }
TEST(ScalarTest, ACosFloatSecondOrder) { test_acos<float, true>(1e-4f); }
TEST(ScalarTest, ACosDoubleSecondOrder) { test_acos<double, true>(1e-12); }
TEST(ScalarTest, ACosLongDoubleSecondOrder) { test_acos<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_atan(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const auto f = atan(a);
    ASSERT_NEAR(f.val, std::atan(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 2.4, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -4.16, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ATanFloatFirstOrder) { test_atan<float, false>(1e-4f); }
TEST(ScalarTest, ATanDoubleFirstOrder) { test_atan<double, false>(1e-12); }
TEST(ScalarTest, ATanLongDoubleFirstOrder) { test_atan<long double, false>(1e-12); }
TEST(ScalarTest, ATanFloatSecondOrder) { test_atan<float, true>(1e-4f); }
TEST(ScalarTest, ATanDoubleSecondOrder) { test_atan<double, true>(1e-12); }
TEST(ScalarTest, ATanLongDoubleSecondOrder) { test_atan<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_sinh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = sinh(a);
    ASSERT_NEAR(f.val, std::sinh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::cosh(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 9.0 * std::sinh(4.0) + 2.0 * std::cosh(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, SinhFloatFirstOrder) { test_sinh<float, false>(1e-4f); }
TEST(ScalarTest, SinhDoubleFirstOrder) { test_sinh<double, false>(1e-12); }
TEST(ScalarTest, SinhLongDoubleFirstOrder) { test_sinh<long double, false>(1e-12); }
TEST(ScalarTest, SinhFloatSecondOrder) { test_sinh<float, true>(1e-4f); }
TEST(ScalarTest, SinhDoubleSecondOrder) { test_sinh<double, true>(1e-12); }
TEST(ScalarTest, SinhLongDoubleSecondOrder) { test_sinh<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_cosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = cosh(a);
    ASSERT_NEAR(f.val, std::cosh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 * std::sinh(4.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * std::sinh(4.0) + 9.0 * std::cosh(4.0), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, CoshFloatFirstOrder) { test_cosh<float, false>(1e-4f); }
TEST(ScalarTest, CoshDoubleFirstOrder) { test_cosh<double, false>(1e-12); }
TEST(ScalarTest, CoshLongDoubleFirstOrder) { test_cosh<long double, false>(1e-12); }
TEST(ScalarTest, CoshFloatSecondOrder) { test_cosh<float, true>(1e-4f); }
TEST(ScalarTest, CoshDoubleSecondOrder) { test_cosh<double, true>(1e-12); }
TEST(ScalarTest, CoshLongDoubleSecondOrder) { test_cosh<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_tanh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = tanh(a);
    ASSERT_NEAR(f.val, std::tanh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cosh(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * (1.0 - 9.0 * std::sinh(4.0) / std::cosh(4.0)) / (sqr(std::cosh(4.0))), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, PassiveTanhFloatFirstOrder) { test_tanh<float, false>(1e-4f); }
TEST(ScalarTest, PassiveTanhDoubleFirstOrder) { test_tanh<double, false>(1e-12); }
TEST(ScalarTest, PassiveTanhLongDoubleFirstOrder) { test_tanh<long double, false>(1e-12); }
TEST(ScalarTest, PassiveTanhFloatSecondOrder) { test_tanh<float, true>(1e-4f); }
TEST(ScalarTest, PassiveTanhDoubleSecondOrder) { test_tanh<double, true>(1e-12); }
TEST(ScalarTest, PassiveTanhLongDoubleSecondOrder) { test_tanh<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_asinh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const auto f = asinh(a);
    ASSERT_NEAR(f.val, std::asinh(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 2.68328, 1e-5);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -1.43108, 1e-5);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ASinhFloatFirstOrder) { test_asinh<float, false>(1e-4f); }
TEST(ScalarTest, ASinhDoubleFirstOrder) { test_asinh<double, false>(1e-12); }
TEST(ScalarTest, ASinhLongDoubleFirstOrder) { test_asinh<long double, false>(1e-12); }
TEST(ScalarTest, ASinhFloatSecondOrder) { test_asinh<float, true>(1e-4f); }
TEST(ScalarTest, ASinhDoubleSecondOrder) { test_asinh<double, true>(1e-12); }
TEST(ScalarTest, ASinhLongDoubleSecondOrder) { test_asinh<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_acosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const auto f = acosh(a);
    ASSERT_NEAR(f.val, std::acosh(4), _eps);
    ASSERT_NEAR(f.grad(0), std::sqrt(3.0 / 5.0), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -2.0 / 5.0 / std::sqrt(15), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ACoshFloatFirstOrder) { test_acosh<float, false>(1e-4f); }
TEST(ScalarTest, ACoshDoubleFirstOrder) { test_acosh<double, false>(1e-12); }
TEST(ScalarTest, ACoshLongDoubleFirstOrder) { test_acosh<long double, false>(1e-12); }
TEST(ScalarTest, ACoshFloatSecondOrder) { test_acosh<float, true>(1e-4f); }
TEST(ScalarTest, ACoshDoubleSecondOrder) { test_acosh<double, true>(1e-12); }
TEST(ScalarTest, ACoshLongDoubleSecondOrder) { test_acosh<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_atanh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const auto f = atanh(a);
    ASSERT_NEAR(f.val, std::atanh(0.5), _eps);
    ASSERT_NEAR(f.grad(0), 4.0, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 18.6667, 1e-4);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, ATanhFloatFirstOrder) { test_atanh<float, false>(1e-4f); }
TEST(ScalarTest, ATanhDoubleFirstOrder) { test_atanh<double, false>(1e-12); }
TEST(ScalarTest, ATanhLongDoubleFirstOrder) { test_atanh<long double, false>(1e-12); }
TEST(ScalarTest, ATanhFloatSecondOrder) { test_atanh<float, true>(1e-4f); }
TEST(ScalarTest, ATanhDoubleSecondOrder) { test_atanh<double, true>(1e-12); }
TEST(ScalarTest, ATanhLongDoubleSecondOrder) { test_atanh<long double, true>(1e-12); }

template <typename PassiveT>
void test_isnan_isinf()
{
    TinyAD::Scalar<1, PassiveT> a = 0.0;
    TinyAD::Scalar<1, PassiveT> b = INFINITY;
    TinyAD::Scalar<1, PassiveT> c = -INFINITY;
    TinyAD::Scalar<1, PassiveT> d = NAN;
    ASSERT_EQ(isnan(a), false);
    ASSERT_EQ(isnan(b), false);
    ASSERT_EQ(isnan(c), false);
    ASSERT_EQ(isnan(d), true);
    ASSERT_EQ(isinf(a), false);
    ASSERT_EQ(isinf(b), true);
    ASSERT_EQ(isinf(c), true);
    ASSERT_EQ(isinf(d), false);
    ASSERT_EQ(isfinite(a), true);
    ASSERT_EQ(isfinite(b), false);
    ASSERT_EQ(isfinite(c), false);
    ASSERT_EQ(isfinite(d), false);
}

TEST(ScalarTest, IsnanIsinfFloat) { test_isnan_isinf<float>(); }
TEST(ScalarTest, IsnanIsinfDouble) { test_isnan_isinf<double>(); }
TEST(ScalarTest, IsnanIsinfLongDouble) { test_isnan_isinf<long double>(); }

template <typename PassiveT, bool with_hessian>
void test_plus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

    {   // Test const operator
        const auto f = a + b;
        ASSERT_NEAR(f.val, 4.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 4.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = a + 1.0;
        ASSERT_NEAR(f.val, a.val + 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = 1.0 + a;
        ASSERT_NEAR(f.val, a.val + 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test assignment operator
        a += b;
        ASSERT_NEAR(a.val, 4.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 4.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }

    {   // Test assignment operator double overload
        a += 1.0;
        ASSERT_NEAR(a.val, 5.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 4.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, PlusFloatFirstOrder) { test_plus<float, false>(); }
TEST(ScalarTest, PlusDoubleFirstOrder) { test_plus<double, false>(); }
TEST(ScalarTest, PlusLongDoubleFirstOrder) { test_plus<long double, false>(); }
TEST(ScalarTest, PlusFloatSecondOrder) { test_plus<float, true>(); }
TEST(ScalarTest, PlusDoubleSecondOrder) { test_plus<double, true>(); }
TEST(ScalarTest, PlusLongDoubleSecondOrder) { test_plus<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

    {   // Test const operator
        const auto f = a - b;
        ASSERT_NEAR(f.val, 4.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 2.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), -2.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = a - 1.0;
        ASSERT_NEAR(f.val, a.val - 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = 1.0 - a;
        ASSERT_NEAR(f.val, 1.0 - a.val, 1e-12);
        ASSERT_NEAR(f.grad(0), -a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), -a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test assignment operator
        a -= b;
        ASSERT_NEAR(a.val, 4.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 2.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), -2.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }

    {   // Test assignment operator double overload
        a -= 1.0;
        ASSERT_NEAR(a.val, 3.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 2.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), -2.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, MinusFloatFirstOrder) { test_minus<float, false>(); }
TEST(ScalarTest, MinusDoubleFirstOrder) { test_minus<double, false>(); }
TEST(ScalarTest, MinusLongDoubleFirstOrder) { test_minus<long double, false>(); }
TEST(ScalarTest, MinusFloatSecondOrder) { test_minus<float, true>(); }
TEST(ScalarTest, MinusDoubleSecondOrder) { test_minus<double, true>(); }
TEST(ScalarTest, MinusLongDoubleSecondOrder) { test_minus<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_mult()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

    {   // Test const operator
        const auto f = a * b;
        ASSERT_NEAR(f.val, 0.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 4.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 22.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = a * 2.0;
        ASSERT_NEAR(f.val, 2.0 * a.val, 1e-12);
        ASSERT_NEAR(f.grad(0), 2.0 * a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 2.0 * a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = 2.0 * a;
        ASSERT_NEAR(f.val, 2.0 * a.val, 1e-12);
        ASSERT_NEAR(f.grad(0), 2.0 * a.grad(0), 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 2.0 * a.Hess(0, 0), 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test assignment operator
        a *= b;
        ASSERT_NEAR(a.val, 0.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 4.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 22.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }

    {   // Test assignment operator double overload
        a *= 2.0;
        ASSERT_NEAR(a.val, 0.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 8.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 44.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, MultFloatFirstOrder) { test_mult<float, false>(); }
TEST(ScalarTest, MultDoubleFirstOrder) { test_mult<double, false>(); }
TEST(ScalarTest, MultLongDoubleFirstOrder) { test_mult<long double, false>(); }
TEST(ScalarTest, MultFloatSecondOrder) { test_mult<float, true>(); }
TEST(ScalarTest, MultDoubleSecondOrder) { test_mult<double, true>(); }
TEST(ScalarTest, MultLongDoubleSecondOrder) { test_mult<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_div()
{
    // a(x) = x^3 - x^2 + 1 at x=1
    // b(x) = x^2 + x + 2 at x=1
    TinyAD::Scalar<1, PassiveT, with_hessian> a(1.0, 1.0, 4.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> b(4.0, 3.0, 2.0);

    {   // Test const operator
        const auto f = a / b;
        ASSERT_NEAR(f.val, 1.0 / 4.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 1.0 / 16.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 25.0 / 32.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = a / 2.0;
        ASSERT_NEAR(f.val, a.val / 2.0, 1e-12);
        ASSERT_NEAR(f.grad(0), a.grad(0) / 2.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), a.Hess(0, 0) / 2.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test const operator double overload
        const auto f = 2.0 / a;
        ASSERT_NEAR(f.val, 2.0, 1e-12);
        ASSERT_NEAR(f.grad(0), -2.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), -4.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // Test assignment operator
        a /= b;
        ASSERT_NEAR(a.val, 1.0 / 4.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 1.0 / 16.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 25.0 / 32.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }

    {   // Test assignment operator double overload
        a /= 2.0;
        ASSERT_NEAR(a.val, 1.0 / 8.0, 1e-12);
        ASSERT_NEAR(a.grad(0), 1.0 / 32.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(a.Hess(0, 0), 25.0 / 64.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(a.Hess, 1e-12);
        }
    }
}

TEST(ScalarTest, DivFloatFirstOrder) { test_div<float, false>(); }
TEST(ScalarTest, DivDoubleFirstOrder) { test_div<double, false>(); }
TEST(ScalarTest, DivLongDoubleFirstOrder) { test_div<long double, false>(); }
TEST(ScalarTest, DivFloatSecondOrder) { test_div<float, true>(); }
TEST(ScalarTest, DivDoubleSecondOrder) { test_div<double, true>(); }
TEST(ScalarTest, DivLongDoubleSecondOrder) { test_div<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_atan2_const(const PassiveT _eps)
{
    auto test = [&] (const auto _x, const auto _y)
    {
        TinyAD::Scalar<2, PassiveT, with_hessian> x(_x, 0);
        TinyAD::Scalar<2, PassiveT, with_hessian> y(_y, 1);
        const auto f = atan2(y, x);
        ASSERT_NEAR(f.val, std::atan2(y.val, x.val), _eps);
        ASSERT_NEAR(f.grad(0), -_y / (sqr(_x) + sqr(_y)), _eps);
        ASSERT_NEAR(f.grad(1), _x / (sqr(_x) + sqr(_y)), _eps);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 2.0 * _x * _y / sqr(sqr(_x) + sqr(_y)), _eps);
            ASSERT_NEAR(f.Hess(0, 1), (sqr(_y) - sqr(_x)) / sqr(sqr(_x) + sqr(_y)), _eps);
            ASSERT_NEAR(f.Hess(1, 0), (sqr(_y) - sqr(_x)) / sqr(sqr(_x) + sqr(_y)), _eps);
            ASSERT_NEAR(f.Hess(1, 1), -2.0 * _x * _y / sqr(sqr(_x) + sqr(_y)), _eps);
        }
    };

    test((PassiveT)1.0, (PassiveT)2.0);
    test((PassiveT)2.0, (PassiveT)2.0);
    test((PassiveT)-1.0, (PassiveT)2.0);
    test((PassiveT)-2.0, (PassiveT)3.0);
    test((PassiveT)1.0, (PassiveT)0.0);
    test((PassiveT)0.0, (PassiveT)1.0);
    test((PassiveT)-1.0, (PassiveT)0.0);
    test((PassiveT)0.0, (PassiveT)-1.0);
}

TEST(ScalarTest, Atan2ConstFloatFirstOrder) { test_atan2_const<float, false>(1e-4f); }
TEST(ScalarTest, Atan2ConstDoubleFirstOrder) { test_atan2_const<double, false>(1e-12); }
TEST(ScalarTest, Atan2ConstLongDoubleFirstOrder) { test_atan2_const<long double, false>(1e-12); }
TEST(ScalarTest, Atan2ConstFloatSecondOrder) { test_atan2_const<float, true>(1e-4f); }
TEST(ScalarTest, Atan2ConstDoubleSecondOrder) { test_atan2_const<double, true>(1e-12); }
TEST(ScalarTest, Atan2ConstLongDoubleSecondOrder) { test_atan2_const<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_atan2_1(const PassiveT _eps)
{
    // Test atan2 with 1D curve parametrization
    auto test = [&] (const PassiveT _x)
    {
        TinyAD::Scalar<1, PassiveT, with_hessian> x(_x, 0);

        // Point on parabola
        auto y = sqr(x) - x - 1.0;

        // Polar angle
        auto angle = atan2(y, x);

        // https://www.wolframalpha.com/input/?i=arctan%28%28x%5E2+-+x+-+1%29+%2F+x%29
        if (_x != 0.0)
            ASSERT_NEAR(angle.val, atan2((sqr(_x) - _x - (PassiveT)1.0), _x), _eps);

        // https://www.wolframalpha.com/input/?i=d%2Fdx+arctan%28%28x%5E2+-+x+-+1%29+%2F+x%29
        ASSERT_NEAR(angle.grad(0),
                    (1.0 + sqr(_x)) / (1.0 + 2.0 * _x - 2.0 * sqr(_x) * _x + sqr(_x) * sqr(_x)),
                    _eps);

        if constexpr (with_hessian)
        {
            // https://www.wolframalpha.com/input/?i=d%5E2%2Fdx%5E2+arctan%28%28x%5E2+-+x+-+1%29+%2F+x%29
            ASSERT_NEAR(angle.Hess(0, 0),
                        (2.0 * (-1.0 + _x + 4.0 * sqr(_x) - 2.0 * sqr(_x) * _x + sqr(sqr(_x)) - sqr(sqr(_x)) * _x))
                            / sqr((1.0 + 2.0 * _x - 2.0 * sqr(_x) * _x + sqr(sqr(_x)))),
                        _eps);
        }
    };

    test(-2.0);
    test(-1.0);
    test(-0.5);
    test(-0.25);
    test(0.25);
    test(0.5);
    test(1.0);
    test(2.0);
}

TEST(ScalarTest, Atan2_1FloatFirstOrder) { test_atan2_1<float, false>(1e-4f); }
TEST(ScalarTest, Atan2_1DoubleFirstOrder) { test_atan2_1<double, false>(1e-12); }
TEST(ScalarTest, Atan2_1LongDoubleFirstOrder) { test_atan2_1<long double, false>(1e-12); }
TEST(ScalarTest, Atan2_1FloatSecondOrder) { test_atan2_1<float, true>(1e-4f); }
TEST(ScalarTest, Atan2_1DoubleSecondOrder) { test_atan2_1<double, true>(1e-12); }
TEST(ScalarTest, Atan2_1LongDoubleSecondOrder) { test_atan2_1<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_atan2_2(const PassiveT _eps)
{
    // Test atan2 with distorted 2D parametrization
    auto test = [&] (const PassiveT _x, const PassiveT _y)
    {
        TinyAD::Scalar<2, PassiveT, with_hessian> x(_x, 0);
        TinyAD::Scalar<2, PassiveT, with_hessian> y(_y, 1);

        // Point (a, b)
        auto a = 0.5 * sqr(x) - sqr(y) - y;
        auto b = -sqr(x-2) - sqr(y-3) + 1;

        // Polar angle
        auto angle = atan2(b, a);

        // Compare to atan
        auto angle_ref = atan(b / a);
        ASSERT_NEAR(angle.grad(0), angle_ref.grad(0), _eps);
        ASSERT_NEAR(angle.grad(1), angle_ref.grad(1), _eps);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(angle.Hess(0, 0), angle_ref.Hess(0, 0), _eps);
            ASSERT_NEAR(angle.Hess(0, 1), angle_ref.Hess(0, 1), _eps);
            ASSERT_NEAR(angle.Hess(1, 0), angle_ref.Hess(1, 0), _eps);
            ASSERT_NEAR(angle.Hess(1, 1), angle_ref.Hess(1, 1), _eps);
        }

        // Compare to Maple-generated
        const auto ddx = ((PassiveT) (-2 * _x + 4) / (0.5e0 * (PassiveT) _x * (PassiveT) _x - _y * _y - _y) - 0.10e1 * (-(PassiveT)  pow((PassiveT) (_x - 2), (PassiveT) 2) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * (PassiveT) _x * (PassiveT) _x - _y * _y - _y, -0.2e1) * (PassiveT) _x) / (pow(-(PassiveT)  pow((PassiveT) (_x - 2), (PassiveT) 2) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * (PassiveT) _x * (PassiveT) _x - _y * _y - _y, -0.2e1) + 0.1e1);
        const auto ddy = ((PassiveT) (-2 * _y + 6) / (0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y) - (-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) * (PassiveT) (-2 * _y - 1)) / (pow(-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) + 0.1e1);
        ASSERT_NEAR(angle.grad(0), ddx, _eps);
        ASSERT_NEAR(angle.grad(1), ddy, _eps);
        if constexpr (with_hessian)
        {
            const auto ddxx = (-0.2e1 / (0.5e0 * _x * _x - _y * _y - _y) - 0.20e1 * (-0.2e1 * _x + 0.4e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * _x + 0.200e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.3e1) * _x * _x - 0.10e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1)) / (pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) + 0.1e1) - ((-0.2e1 * _x + 0.4e1) / (0.5e0 * _x * _x - _y * _y - _y) - 0.10e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * _x) * pow(pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) + 0.1e1, -0.2e1) * (0.2e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * (-0.2e1 * _x + 0.4e1) - 0.20e1 * pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.3e1) * _x);
            const auto ddxy = (-0.10e1 * (PassiveT) (-2 * _y + 6) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) * _x - (-0.2e1 * _x + 0.4e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) * (PassiveT) (-2 * _y - 1) + 0.20e1 * (-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.3e1) * (PassiveT) (-2 * _y - 1) * _x) / (pow(-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) + 0.1e1) - ((PassiveT) (-2 * _y + 6) / (0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y) - (-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) * (PassiveT) (-2 * _y - 1)) * pow(pow(-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) + 0.1e1, -0.2e1) * (0.2e1 * (-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.2e1) * (-0.2e1 * _x + 0.4e1) - 0.20e1 * pow(-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - (PassiveT) (_y * _y) - (PassiveT) _y, -0.3e1) * _x);
            const auto ddyy = (-0.2e1 / (0.5e0 * _x * _x - _y * _y - _y) - 0.2e1 * (-0.2e1 * _y + 0.6e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * (-0.2e1 * _y - 0.1e1) + 0.2e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.3e1) * pow(-0.2e1 * _y - 0.1e1, 0.2e1) + 0.2e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1)) / (pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) + 0.1e1) - ((-0.2e1 * _y + 0.6e1) / (0.5e0 * _x * _x - _y * _y - _y) - (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * (-0.2e1 * _y - 0.1e1)) * pow(pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) + 0.1e1, -0.2e1) * (0.2e1 * (-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.2e1) * (-0.2e1 * _y + 0.6e1) - 0.2e1 * pow(-pow(_x - 0.2e1, 0.2e1) - pow(_y - 0.3e1, 0.2e1) + 0.1e1, 0.2e1) * pow(0.5e0 * _x * _x - _y * _y - _y, -0.3e1) * (-0.2e1 * _y - 0.1e1));
            ASSERT_NEAR(angle.Hess(0, 0), ddxx, _eps);
            ASSERT_NEAR(angle.Hess(0, 1), ddxy, _eps);
            ASSERT_NEAR(angle.Hess(1, 0), ddxy, _eps);
            ASSERT_NEAR(angle.Hess(1, 1), ddyy, _eps);
        }
    };

    test(1.0, 0.0);
    test(0.5, 0.5);
    test(0.0, 1.0);
    test(-0.5, 0.5);
    test(-1.0, 0.0);
    test(-0.5, -0.5);
    test(0.0, 1.0);
    test(0.5, -0.5);
}

TEST(ScalarTest, Atan2_2FloatFirstOrder) { test_atan2_2<float, false>(1e-4f); }
TEST(ScalarTest, Atan2_2DoubleFirstOrder) { test_atan2_2<double, false>(1e-12); }
TEST(ScalarTest, Atan2_2LongDoubleFirstOrder) { test_atan2_2<long double, false>(1e-12); }
TEST(ScalarTest, Atan2_2FloatSecondOrder) { test_atan2_2<float, true>(1e-4f); }
TEST(ScalarTest, Atan2_2DoubleSecondOrder) { test_atan2_2<double, true>(1e-12); }
TEST(ScalarTest, Atan2_2LongDoubleSecondOrder) { test_atan2_2<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_hypot(const PassiveT _eps)
{
    TinyAD::Scalar<2, PassiveT, with_hessian> x(3.0, 0);
    TinyAD::Scalar<2, PassiveT, with_hessian> y(4.0, 1);
    TinyAD::Scalar<2, PassiveT, with_hessian> z = hypot(x, y);

    ASSERT_NEAR(z.val, 5.0, _eps);

    ASSERT_NEAR(z.grad(0), 3.0 / 5.0, _eps);
    ASSERT_NEAR(z.grad(1), 4.0 / 5.0, _eps);

    if constexpr (with_hessian)
    {
        ASSERT_NEAR(z.Hess(0, 0), 16.0 / 25.0 / 5.0, _eps);
        ASSERT_NEAR(z.Hess(0, 1), -12.0 / 25.0 / 5.0, _eps);
        ASSERT_NEAR(z.Hess(1, 0), -12.0 / 25.0 / 5.0, _eps);
        ASSERT_NEAR(z.Hess(1, 1), 9.0 / 25.0 / 5.0, _eps);
    }
}

TEST(ScalarTest, HypotFloatFirstOrder) { test_hypot<float, false>(1e-7f); }
TEST(ScalarTest, HypotDoubleFirstOrder) { test_hypot<double, false>(1e-12); }
TEST(ScalarTest, HypotLongDoubleFirstOrder) { test_hypot<long double, false>(1e-14); }
TEST(ScalarTest, HypotFloatSecondOrder) { test_hypot<float, true>(1e-7f); }
TEST(ScalarTest, HypotDoubleSecondOrder) { test_hypot<double, true>(1e-12); }
TEST(ScalarTest, HypotLongDoubleSecondOrder) { test_hypot<long double, true>(1e-14); }

template <typename PassiveT, bool with_hessian>
void test_div2d(const PassiveT _eps)
{
    // wolframalpha.com/input/?i=x%5E2+%2F+y
    TinyAD::Scalar<2, PassiveT, with_hessian> x(-1.0, 0);
    TinyAD::Scalar<2, PassiveT, with_hessian> y(-0.5, 1);
    const auto f = sqr(x) / y;
    ASSERT_NEAR(f.val, -2.0, _eps);
    ASSERT_NEAR(f.grad(0), 4.0, _eps);
    ASSERT_NEAR(f.grad(1), -4.0, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), -4.0, _eps);
        ASSERT_NEAR(f.Hess(0, 1), 8.0, _eps);
        ASSERT_NEAR(f.Hess(1, 0), 8.0, _eps);
        ASSERT_NEAR(f.Hess(1, 1), -16.0, _eps);
    }
}

TEST(ScalarTest, Div2dFloatFirstOrder) { test_div2d<float, false>(1e-4f); }
TEST(ScalarTest, Div2dDoubleFirstOrder) { test_div2d<double, false>(1e-12); }
TEST(ScalarTest, Div2dLongDoubleFirstOrder) { test_div2d<long double, false>(1e-12); }
TEST(ScalarTest, Div2dFloatSecondOrder) { test_div2d<float, true>(1e-4f); }
TEST(ScalarTest, Div2dDoubleSecondOrder) { test_div2d<double, true>(1e-12); }
TEST(ScalarTest, Div2dLongDoubleSecondOrder) { test_div2d<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_div2d_2(const PassiveT _eps)
{
    auto test = [&] (const PassiveT _x, const PassiveT _y)
    {
        TinyAD::Scalar<2, PassiveT, with_hessian> x(_x, 0);
        TinyAD::Scalar<2, PassiveT, with_hessian> y(_y, 1);

        auto a = 0.5 * sqr(x) - sqr(y) + 2.0 * x - y;
        auto b = -sqr(x - 2.0) - sqr(y - 3.0) + 1.0;
        if constexpr (with_hessian)
        {
            TINYAD_ASSERT_SYMMETRIC(a.Hess, _eps);
            TINYAD_ASSERT_SYMMETRIC(b.Hess, _eps);
        }

        auto f = a / b;

        // Compare to Maple-generated
        const double dx = (0.10e1 * _x + 0.2e1) / (-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1) - (0.5e0 * _x * _x - (PassiveT) (_y * _y) + 0.2e1 * _x - (PassiveT) _y) * pow(-pow(_x - 0.2e1, 0.2e1) - (PassiveT)  pow((PassiveT) (_y - 3), (PassiveT) 2) + 0.1e1, -0.2e1) * (-0.2e1 * _x + 0.4e1);
        const double dy = (PassiveT) ((-2 * _y - 1) / (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1)) - (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2)) * (PassiveT) (-2 * _y + 6);
        ASSERT_NEAR(f.grad(0), dx, _eps);
        ASSERT_NEAR(f.grad(1), dy, _eps);
        if constexpr (with_hessian)
        {
            const double dxx = 0.10e1 / (PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1) - 0.2e1 * (0.10e1 * (PassiveT) _x + 0.2e1) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2)) * (PassiveT) (-2 * _x + 4) + 0.2e1 * (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-3)) * (PassiveT)  pow((PassiveT) (-2 * _x + 4), (PassiveT) 2) + 0.2e1 * (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2));
            const double dxy = -(PassiveT) ((-2 * _y - 1) *  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2)) * (-2 * _x + 4)) - (0.10e1 * (PassiveT) _x + 0.2e1) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2)) * (PassiveT) (-2 * _y + 6) + 0.2e1 * (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-3)) * (PassiveT) (-2 * _y + 6) * (PassiveT) (-2 * _x + 4);
            const double dyy = -(PassiveT) (2 / (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1)) - (PassiveT) (2 * (-2 * _y - 1) *  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2)) * (-2 * _y + 6)) + 0.2e1 * (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-3)) * (PassiveT)  pow((PassiveT) (-2 * _y + 6), (PassiveT) 2) + 0.2e1 * (0.5e0 * (PassiveT) _x * (PassiveT) _x - (PassiveT) (_y * _y) + (PassiveT) (2 * _x) - (PassiveT) _y) * (PassiveT)  pow((PassiveT) (- pow((PassiveT) (_x - 2), (PassiveT) 2) -  pow((PassiveT) (_y - 3), (PassiveT) 2) + 1), (PassiveT) (-2));
            ASSERT_NEAR(f.Hess(0, 0), dxx, _eps);
            ASSERT_NEAR(f.Hess(0, 1), dxy, _eps);
            ASSERT_NEAR(f.Hess(1, 0), dxy, _eps);
            ASSERT_NEAR(f.Hess(1, 1), dyy, _eps);
        }
    };

    test(5.0, 0.0);
    test(1.0, 1.0);
    test(0.0, 5.0);
    test(-1.0, 1.0);
    test(-5.0, 0.0);
    test(-1.0, -1.0);
    test(0.0, -5.0);
    test(1.0, -1.0);
}

TEST(ScalarTest, Div2d_2FloatFirstOrder) { test_div2d_2<float, false>(1e-4f); }
TEST(ScalarTest, Div2d_2DoubleFirstOrder) { test_div2d_2<double, false>(1e-12); }
TEST(ScalarTest, Div2d_2LongDoubleFirstOrder) { test_div2d_2<long double, false>(1e-12); }
TEST(ScalarTest, Div2d_2FloatSecondOrder) { test_div2d_2<float, true>(1e-4f); }
TEST(ScalarTest, Div2d_2DoubleSecondOrder) { test_div2d_2<double, true>(1e-12); }
TEST(ScalarTest, Div2d_2LongDoubleSecondOrder) { test_div2d_2<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_plus_minus_mult_div_2d(const PassiveT _eps)
{
    // wolframalpha.com/input/?i=%28%28x%5E2%2Bx%29+*+%28y%5E2-y%29+%2F+%28y-1%29%29+
    TinyAD::Scalar<2, PassiveT, with_hessian> x(1.0, 0);
    TinyAD::Scalar<2, PassiveT, with_hessian> y(1.5, 1);
    const auto f = (sqr(x) + x) * (sqr(y) - y) / (y - 1.0);
    ASSERT_NEAR(f.val, 3.0, _eps);
    ASSERT_NEAR(f.grad(0), 4.5, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 3.0, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, PlusMinusMultDiv2dFloatFirstOrder) { test_plus_minus_mult_div_2d<float, false>(1e-4f); }
TEST(ScalarTest, PlusMinusMultDiv2dDoubleFirstOrder) { test_plus_minus_mult_div_2d<double, false>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dLongDoubleFirstOrder) { test_plus_minus_mult_div_2d<long double, false>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dFloatSecondOrder) { test_plus_minus_mult_div_2d<float, true>(1e-4f); }
TEST(ScalarTest, PlusMinusMultDiv2dDoubleSecondOrder) { test_plus_minus_mult_div_2d<double, true>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dLongDoubleSecondOrder) { test_plus_minus_mult_div_2d<long double, true>(1e-12); }

template <typename PassiveT, bool with_hessian>
void test_comparison()
{
    TinyAD::Scalar<1, PassiveT, with_hessian> a(1.0, 1.0, 4.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> b(1.0, 2.0, 8.0);
    TinyAD::Scalar<1, PassiveT, with_hessian> c(2.0, 2.0, 8.0);

    ASSERT_TRUE(a == b);
    ASSERT_TRUE(b == a);
    ASSERT_TRUE(a != c);
    ASSERT_TRUE(c != a);
    ASSERT_TRUE(b != c);
    ASSERT_TRUE(c != b);

    ASSERT_FALSE(a < b);
    ASSERT_FALSE(b < a);
    ASSERT_TRUE(a < c);
    ASSERT_FALSE(c < a);
    ASSERT_TRUE(b < c);
    ASSERT_FALSE(c < b);

    ASSERT_TRUE(a <= b);
    ASSERT_TRUE(b <= a);
    ASSERT_TRUE(a <= c);
    ASSERT_FALSE(c <= a);
    ASSERT_TRUE(b <= c);
    ASSERT_FALSE(c <= b);

    ASSERT_FALSE(a > b);
    ASSERT_FALSE(b > a);
    ASSERT_FALSE(a > c);
    ASSERT_TRUE(c > a);
    ASSERT_FALSE(b > c);
    ASSERT_TRUE(c > b);

    ASSERT_TRUE(a >= b);
    ASSERT_TRUE(b >= a);
    ASSERT_FALSE(a >= c);
    ASSERT_TRUE(c >= a);
    ASSERT_FALSE(b >= c);
    ASSERT_TRUE(c >= b);

    // Test double overloads
    ASSERT_TRUE(a == 1.0);
    ASSERT_FALSE(a == 2.0);
    ASSERT_FALSE(a != 1.0);
    ASSERT_TRUE(a != 2.0);
    ASSERT_FALSE(a < 1.0);
    ASSERT_TRUE(a < 2.0);
    ASSERT_TRUE(a <= 1.0);
    ASSERT_TRUE(a <= 2.0);
    ASSERT_FALSE(a > 1.0);
    ASSERT_FALSE(a > 2.0);
    ASSERT_TRUE(a >= 1.0);
    ASSERT_FALSE(a >= 2.0);
}

TEST(ScalarTest, ComparisonFloatFirstOrder) { test_comparison<float, false>(); }
TEST(ScalarTest, ComparisonDoubleFirstOrder) { test_comparison<double, false>(); }
TEST(ScalarTest, ComparisonLongDoubleFirstOrder) { test_comparison<long double, false>(); }
TEST(ScalarTest, ComparisonFloatSecondOrder) { test_comparison<float, true>(); }
TEST(ScalarTest, ComparisonDoubleSecondOrder) { test_comparison<double, true>(); }
TEST(ScalarTest, ComparisonLongDoubleSecondOrder) { test_comparison<long double, true>(); }

template <typename PassiveT>
void test_min_max()
{
    TinyAD::Scalar<1, PassiveT> a = 1.0;
    TinyAD::Scalar<1, PassiveT> b = 2.0;

    ASSERT_EQ(min(a, b), a);
    ASSERT_EQ(min(a, b).grad, a.grad);
    ASSERT_EQ(min(a, b).Hess, a.Hess);

    ASSERT_EQ(fmin(a, b), a);
    ASSERT_EQ(fmin(a, b).grad, a.grad);
    ASSERT_EQ(fmin(a, b).Hess, a.Hess);

    ASSERT_EQ(max(a, b), b);
    ASSERT_EQ(max(a, b).grad, b.grad);
    ASSERT_EQ(max(a, b).Hess, b.Hess);

    ASSERT_EQ(fmax(a, b), b);
    ASSERT_EQ(fmax(a, b).grad, b.grad);
    ASSERT_EQ(fmax(a, b).Hess, b.Hess);
}

TEST(ScalarTest, MinMaxFloat) { test_min_max<float>(); }
TEST(ScalarTest, MinMaxDouble) { test_min_max<double>(); }
TEST(ScalarTest, MinMaxLongDouble) { test_min_max<long double>(); }

template <typename PassiveT>
void test_clamp()
{
    TinyAD::Scalar<1, PassiveT, with_hessian> x(4.0, 3.0, 2.0);

    ASSERT_EQ(clamp(x, 0.0, 5.0), x);
    ASSERT_EQ(clamp(x, 0.0, 5.0).grad, x.grad);
    ASSERT_EQ(clamp(x, 0.0, 5.0).Hess, x.Hess);

    ASSERT_EQ(clamp(x, -5.0, 0.0), 0.0);
    ASSERT_EQ(clamp(x, -5.0, 0.0).grad(0), 0.0);
    ASSERT_EQ(clamp(x, -5.0, 0.0).Hess(0, 0), 0.0);

    ASSERT_EQ(clamp(x, 5.0, 10.0), 5.0);
    ASSERT_EQ(clamp(x, 5.0, 10.0).grad(0), 0.0);
    ASSERT_EQ(clamp(x, 5.0, 10.0).Hess(0, 0), 0.0);
}

TEST(ScalarTest, MinMaxFloat) { test_clamp<float>(); }
TEST(ScalarTest, MinMaxDouble) { test_clamp<double>(); }
TEST(ScalarTest, MinMaxLongDouble) { test_clamp<long double>(); }

template <typename PassiveT, bool with_hessian>
void test_sphere()
{
    // f: R^2 -> R^3
    // f(phi, psi) = (sin(phi) * cos(psi), sin(phi) * sin(psi), cos(phi))
    TinyAD::Scalar<2, PassiveT, with_hessian> alpha((PassiveT)M_PI / 8.0, 0);
    TinyAD::Scalar<2, PassiveT, with_hessian> beta((PassiveT)M_PI / 8.0, 1);
    const auto f = Eigen::Matrix<TinyAD::Scalar<2, PassiveT, with_hessian>, 3, 1>(
            sin(alpha) * cos(beta),
            sin(alpha) * sin(beta),
            cos(alpha));

    // Test function value
    ASSERT_NEAR(f[0].val, std::sin(alpha.val) * std::cos(alpha.val), 1e-12);
    ASSERT_NEAR(f[1].val, std::sin(alpha.val) * std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].val, std::cos(alpha.val), 1e-12);

    // Test gradient (Jacobian)
    ASSERT_NEAR(f[0].grad(0), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
    ASSERT_NEAR(f[0].grad(1), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
    ASSERT_NEAR(f[1].grad(0), std::cos(alpha.val) * std::sin(beta.val), 1e-12);
    ASSERT_NEAR(f[1].grad(1), std::cos(beta.val) * std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].grad(0), -std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].grad(1), 0.0, 1e-12);

    if constexpr (with_hessian)
    {
        // Test Hessian
        ASSERT_NEAR(f[0].Hess(0, 0), -std::sin(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(0, 1), -std::cos(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(1, 0), -std::cos(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(1, 1), -std::sin(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(0, 0), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(0, 1), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(1, 0), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(1, 1), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[2].Hess(0, 0), -std::cos(alpha.val), 1e-12);
        ASSERT_NEAR(f[2].Hess(0, 1), 0.0, 1e-12);
        ASSERT_NEAR(f[2].Hess(1, 0), 0.0, 1e-12);
        ASSERT_NEAR(f[2].Hess(1, 1), 0.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[0].Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[1].Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[2].Hess, 1e-12);
    }
}

TEST(ScalarTest, SphereFloatFirstOrder) { test_sphere<float, false>(); }
TEST(ScalarTest, SphereDoubleFirstOrder) { test_sphere<double, false>(); }
TEST(ScalarTest, SphereLongDoubleFirstOrder) { test_sphere<long double, false>(); }
TEST(ScalarTest, SphereFloatSecondOrder) { test_sphere<float, true>(); }
TEST(ScalarTest, SphereDoubleSecondOrder) { test_sphere<double, true>(); }
TEST(ScalarTest, SphereLongDoubleSecondOrder) { test_sphere<long double, true>(); }

template <typename PassiveT>
void test_min_quadric()
{
    // Variable vector in R^3
    const auto x = TinyAD::Scalar<3, PassiveT>::make_active({ 0.0, 0.0, 0.0 });

    // Quadratic function
    const auto f = sqr(x[0]) + 2.0 * sqr(x[1]) + 6.0 * sqr(x[2]) + x[0] - 2.0 * x[1] + 6.0 * x[2] + 10;

    // Solve for minimum
    const auto x_min = -f.Hess.inverse() * f.grad;

    ASSERT_NEAR(x_min.x(), -0.5, 1e-12);
    ASSERT_NEAR(x_min.y(), 0.5, 1e-12);
    ASSERT_NEAR(x_min.z(), -0.5, 1e-12);
}

TEST(ScalarTest, MinQuadraticFloat) { test_min_quadric<float>(); }
TEST(ScalarTest, MinQuadraticDouble) { test_min_quadric<double>(); }
TEST(ScalarTest, MinQuadraticLongDouble) { test_min_quadric<long double>(); }

template <typename PassiveT>
void test_triangle_distortion()
{
    using ad = TinyAD::Scalar<6, PassiveT>;

    // Passive rest-state triangle ar, br, cr
    const Eigen::Matrix<PassiveT, 2, 1> ar(1.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> br(2.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> cr(1.0, 2.0);
    const auto Mr = TinyAD::col_mat(br - ar, cr - ar);

    // Active variable vector for vertex positions a, b, c
    auto x = ad::make_active({
        10.0, 1.0,
        15.0, 3.0,
        2.0, 2.0,
    });
    const Eigen::Matrix<ad, 2, 1> a(x[0], x[1]);
    const Eigen::Matrix<ad, 2, 1> b(x[2], x[3]);
    const Eigen::Matrix<ad, 2, 1> c(x[4], x[5]);
    auto M = TinyAD::col_mat(b - a, c - a);

    const auto J = M * Mr.inverse();
    const auto E = J.squaredNorm() + J.inverse().squaredNorm();
    TINYAD_ASSERT_FINITE(E.val);
    TINYAD_ASSERT_FINITE_MAT(E.grad);
    TINYAD_ASSERT_FINITE_MAT(E.Hess);
}

TEST(ScalarTest, PassiveTriangleDistortionFloat) { test_triangle_distortion<float>(); }
TEST(ScalarTest, PassiveTriangleDistortionDouble) { test_triangle_distortion<double>(); }
TEST(ScalarTest, PassiveTriangleDistortionLongDouble) { test_triangle_distortion<long double>(); }
