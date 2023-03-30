/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_pow_int()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = pow(a, 3);
    ASSERT_NEAR(f.val, 64.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 144.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 312.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTestBinaryOperators, PowFloatIntFirstOrder) { test_pow_int<float, false>(); }
TEST(ScalarTestBinaryOperators, PowDoubleIntFirstOrder) { test_pow_int<double, false>(); }
TEST(ScalarTestBinaryOperators, PowLongDoubleIntFirstOrder) { test_pow_int<long double, false>(); }
TEST(ScalarTestBinaryOperators, PowFloatIntSecondOrder) { test_pow_int<float, true>(); }
TEST(ScalarTestBinaryOperators, PowDoubleIntSecondOrder) { test_pow_int<double, true>(); }
TEST(ScalarTestBinaryOperators, PowLongDoubleIntSecondOrder) { test_pow_int<long double, true>(); }

TEST(ScalarTestBinaryOperators, PowDoubleFirstOrderDynamic) { test_pow_int<double, false, true>(); }
TEST(ScalarTestBinaryOperators, PowDoubleSecondOrderDynamic) { test_pow_int<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_pow_real()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    const ADouble f = pow(a, PassiveT(3.0 / 2.0));
    ASSERT_NEAR(f.val, 8.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 9.0, 1e-12);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 75.0 / 8.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTestBinaryOperators, PowFloatRealFirstOrder) { test_pow_real<float, false>(); }
TEST(ScalarTestBinaryOperators, PowDoubleRealFirstOrder) { test_pow_real<double, false>(); }
TEST(ScalarTestBinaryOperators, PowLongDoubleRealFirstOrder) { test_pow_real<long double, false>(); }
TEST(ScalarTestBinaryOperators, PowFloatRealSecondOrder) { test_pow_real<float, true>(); }
TEST(ScalarTestBinaryOperators, PowDoubleRealSecondOrder) { test_pow_real<double, true>(); }
TEST(ScalarTestBinaryOperators, PowLongDoubleRealSecondOrder) { test_pow_real<long double, true>(); }

TEST(ScalarTestBinaryOperators, PowDoubleRealFirstOrderDynamic) { test_pow_real<double, false, true>(); }
TEST(ScalarTestBinaryOperators, PowDoubleRealSecondOrderDynamic) { test_pow_real<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_plus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    ADouble b = ADouble::known_derivatives(0.0, 1.0, 4.0);

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

TEST(ScalarTestBinaryOperators, PlusFloatFirstOrder) { test_plus<float, false>(); }
TEST(ScalarTestBinaryOperators, PlusDoubleFirstOrder) { test_plus<double, false>(); }
TEST(ScalarTestBinaryOperators, PlusLongDoubleFirstOrder) { test_plus<long double, false>(); }
TEST(ScalarTestBinaryOperators, PlusFloatSecondOrder) { test_plus<float, true>(); }
TEST(ScalarTestBinaryOperators, PlusDoubleSecondOrder) { test_plus<double, true>(); }
TEST(ScalarTestBinaryOperators, PlusLongDoubleSecondOrder) { test_plus<long double, true>(); }

TEST(ScalarTestBinaryOperators, PlusDoubleFirstOrderDynamic) { test_plus<double, false, true>(); }
TEST(ScalarTestBinaryOperators, PlusDoubleSecondOrderDynamic) { test_plus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    ADouble b = ADouble::known_derivatives(0.0, 1.0, 4.0);

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

TEST(ScalarTestBinaryOperators, MinusFloatFirstOrder) { test_minus<float, false>(); }
TEST(ScalarTestBinaryOperators, MinusDoubleFirstOrder) { test_minus<double, false>(); }
TEST(ScalarTestBinaryOperators, MinusLongDoubleFirstOrder) { test_minus<long double, false>(); }
TEST(ScalarTestBinaryOperators, MinusFloatSecondOrder) { test_minus<float, true>(); }
TEST(ScalarTestBinaryOperators, MinusDoubleSecondOrder) { test_minus<double, true>(); }
TEST(ScalarTestBinaryOperators, MinusLongDoubleSecondOrder) { test_minus<long double, true>(); }

TEST(ScalarTestBinaryOperators, MinusDoubleFirstOrderDynamic) { test_minus<double, false, true>(); }
TEST(ScalarTestBinaryOperators, MinusDoubleSecondOrderDynamic) { test_minus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_mult()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    ADouble b = ADouble::known_derivatives(0.0, 1.0, 4.0);

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

TEST(ScalarTestBinaryOperators, MultFloatFirstOrder) { test_mult<float, false>(); }
TEST(ScalarTestBinaryOperators, MultDoubleFirstOrder) { test_mult<double, false>(); }
TEST(ScalarTestBinaryOperators, MultLongDoubleFirstOrder) { test_mult<long double, false>(); }
TEST(ScalarTestBinaryOperators, MultFloatSecondOrder) { test_mult<float, true>(); }
TEST(ScalarTestBinaryOperators, MultDoubleSecondOrder) { test_mult<double, true>(); }
TEST(ScalarTestBinaryOperators, MultLongDoubleSecondOrder) { test_mult<long double, true>(); }

TEST(ScalarTestBinaryOperators, MultDoubleFirstOrderDynamic) { test_mult<double, false, true>(); }
TEST(ScalarTestBinaryOperators, MultDoubleSecondOrderDynamic) { test_mult<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_div()
{
    // a(x) = x^3 - x^2 + 1 at x=1
    // b(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble a = ADouble::known_derivatives(1.0, 1.0, 4.0);
    ADouble b = ADouble::known_derivatives(4.0, 3.0, 2.0);

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

TEST(ScalarTestBinaryOperators, DivFloatFirstOrder) { test_div<float, false>(); }
TEST(ScalarTestBinaryOperators, DivDoubleFirstOrder) { test_div<double, false>(); }
TEST(ScalarTestBinaryOperators, DivLongDoubleFirstOrder) { test_div<long double, false>(); }
TEST(ScalarTestBinaryOperators, DivFloatSecondOrder) { test_div<float, true>(); }
TEST(ScalarTestBinaryOperators, DivDoubleSecondOrder) { test_div<double, true>(); }
TEST(ScalarTestBinaryOperators, DivLongDoubleSecondOrder) { test_div<long double, true>(); }

TEST(ScalarTestBinaryOperators, DivDoubleFirstOrderDynamic) { test_div<double, false, true>(); }
TEST(ScalarTestBinaryOperators, DivDoubleSecondOrderDynamic) { test_div<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atan2_const(const PassiveT _eps)
{
    auto test = [&] (const auto _x, const auto _y)
    {
        constexpr int dim = dynamic ? Eigen::Dynamic : 2;
        using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
        ADouble x = ADouble::make_active(_x, 0, 2);
        ADouble y = ADouble::make_active(_y, 1, 2);
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

TEST(ScalarTestBinaryOperators, Atan2ConstFloatFirstOrder) { test_atan2_const<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2ConstDoubleFirstOrder) { test_atan2_const<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2ConstLongDoubleFirstOrder) { test_atan2_const<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2ConstFloatSecondOrder) { test_atan2_const<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2ConstDoubleSecondOrder) { test_atan2_const<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2ConstLongDoubleSecondOrder) { test_atan2_const<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, Atan2ConstDoubleFirstOrderDynamic) { test_atan2_const<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2ConstDoubleSecondOrderDynamic) { test_atan2_const<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atan2_1(const PassiveT _eps)
{
    // Test atan2 with 1D curve parametrization
    auto test = [&] (const PassiveT _x)
    {
        constexpr int dim = dynamic ? Eigen::Dynamic : 1;
        using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
        ADouble x = ADouble::make_active(_x, 0, 1);

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

TEST(ScalarTestBinaryOperators, Atan2_1FloatFirstOrder) { test_atan2_1<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2_1DoubleFirstOrder) { test_atan2_1<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_1LongDoubleFirstOrder) { test_atan2_1<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_1FloatSecondOrder) { test_atan2_1<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2_1DoubleSecondOrder) { test_atan2_1<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_1LongDoubleSecondOrder) { test_atan2_1<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, Atan2_1DoubleFirstOrderDynamic) { test_atan2_1<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_1DoubleSecondOrderDynamic) { test_atan2_1<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atan2_2(const PassiveT _eps)
{
    // Test atan2 with distorted 2D parametrization
    auto test = [&] (const PassiveT _x, const PassiveT _y)
    {
        constexpr int dim = dynamic ? Eigen::Dynamic : 2;
        using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
        ADouble x = ADouble::make_active(_x, 0, 2);
        ADouble y = ADouble::make_active(_y, 1, 2);

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

TEST(ScalarTestBinaryOperators, Atan2_2FloatFirstOrder) { test_atan2_2<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2_2DoubleFirstOrder) { test_atan2_2<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_2LongDoubleFirstOrder) { test_atan2_2<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_2FloatSecondOrder) { test_atan2_2<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, Atan2_2DoubleSecondOrder) { test_atan2_2<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_2LongDoubleSecondOrder) { test_atan2_2<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, Atan2_2ConstDoubleFirstOrderDynamic) { test_atan2_2<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Atan2_2ConstDoubleSecondOrderDynamic) { test_atan2_2<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_hypot(const PassiveT _eps)
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble x = ADouble::make_active(3.0, 0, 2);
    ADouble y = ADouble::make_active(4.0, 1, 2);
    ADouble z = hypot(x, y);

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

TEST(ScalarTestBinaryOperators, HypotFloatFirstOrder) { test_hypot<float, false>(1e-7f); }
TEST(ScalarTestBinaryOperators, HypotDoubleFirstOrder) { test_hypot<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, HypotLongDoubleFirstOrder) { test_hypot<long double, false>(1e-14); }
TEST(ScalarTestBinaryOperators, HypotFloatSecondOrder) { test_hypot<float, true>(1e-7f); }
TEST(ScalarTestBinaryOperators, HypotDoubleSecondOrder) { test_hypot<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, HypotLongDoubleSecondOrder) { test_hypot<long double, true>(1e-14); }

TEST(ScalarTestBinaryOperators, HypotDoubleFirstOrderDynamic) { test_hypot<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, HypotDoubleSecondOrderDynamic) { test_hypot<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_div2d(const PassiveT _eps)
{
    // wolframalpha.com/input/?i=x%5E2+%2F+y
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble x = ADouble::make_active(-1.0, 0, 2);
    ADouble y = ADouble::make_active(-0.5, 1, 2);

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

TEST(ScalarTestBinaryOperators, Div2dFloatFirstOrder) { test_div2d<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, Div2dDoubleFirstOrder) { test_div2d<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2dLongDoubleFirstOrder) { test_div2d<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2dFloatSecondOrder) { test_div2d<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, Div2dDoubleSecondOrder) { test_div2d<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2dLongDoubleSecondOrder) { test_div2d<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, Div2dDoubleFirstOrderDynamic) { test_div2d<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2dDoubleSecondOrderDynamic) { test_div2d<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_div2d_2(const PassiveT _eps)
{
    auto test = [&] (const PassiveT _x, const PassiveT _y)
    {
        constexpr int dim = dynamic ? Eigen::Dynamic : 2;
        using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
        ADouble x = ADouble::make_active(_x, 0, 2);
        ADouble y = ADouble::make_active(_y, 1, 2);

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

TEST(ScalarTestBinaryOperators, Div2d_2FloatFirstOrder) { test_div2d_2<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, Div2d_2DoubleFirstOrder) { test_div2d_2<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2d_2LongDoubleFirstOrder) { test_div2d_2<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2d_2FloatSecondOrder) { test_div2d_2<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, Div2d_2DoubleSecondOrder) { test_div2d_2<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2d_2LongDoubleSecondOrder) { test_div2d_2<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, Div2d_2DoubleFirstOrderDynamic) { test_div2d_2<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, Div2d_2DoubleSecondOrderDynamic) { test_div2d_2<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_plus_minus_mult_div_2d(const PassiveT _eps)
{
    // wolframalpha.com/input/?i=%28%28x%5E2%2Bx%29+*+%28y%5E2-y%29+%2F+%28y-1%29%29+
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble x = ADouble::make_active(1.0, 0, 2);
    ADouble y = ADouble::make_active(1.5, 1, 2);
    const auto f = (sqr(x) + x) * (sqr(y) - y) / (y - 1.0);
    ASSERT_NEAR(f.val, 3.0, _eps);
    ASSERT_NEAR(f.grad(0), 4.5, _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 3.0, _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dFloatFirstOrder) { test_plus_minus_mult_div_2d<float, false>(1e-4f); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dDoubleFirstOrder) { test_plus_minus_mult_div_2d<double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dLongDoubleFirstOrder) { test_plus_minus_mult_div_2d<long double, false>(1e-12); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dFloatSecondOrder) { test_plus_minus_mult_div_2d<float, true>(1e-4f); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dDoubleSecondOrder) { test_plus_minus_mult_div_2d<double, true>(1e-12); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dLongDoubleSecondOrder) { test_plus_minus_mult_div_2d<long double, true>(1e-12); }

TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dDoubleFirstOrderDynamic) { test_plus_minus_mult_div_2d<double, false, true>(1e-12); }
TEST(ScalarTestBinaryOperators, PlusMinusMultDiv2dDoubleSecondOrderDynamic) { test_plus_minus_mult_div_2d<double, true, true>(1e-12); }
