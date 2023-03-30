/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool dynamic = false>
void test_isnan_isinf()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT>;
    const ADouble a = ADouble::make_passive(0.0, 1);
    const ADouble b = ADouble::make_passive(INFINITY, 1);
    const ADouble c = ADouble::make_passive(-INFINITY, 1);
    const ADouble d = ADouble::make_passive(NAN, 1);
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

TEST(ScalarTestComparison, IsnanIsinfFloat) { test_isnan_isinf<float>(); }
TEST(ScalarTestComparison, IsnanIsinfDouble) { test_isnan_isinf<double>(); }
TEST(ScalarTestComparison, IsnanIsinfLongDouble) { test_isnan_isinf<long double>(); }

TEST(ScalarTestComparison, IsnanIsinfDoubleDynamic) { test_isnan_isinf<double, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_comparison()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT, with_hessian> a(1.0, 1.0, 4.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> b(1.0, 2.0, 8.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> c(2.0, 2.0, 8.0);

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

TEST(ScalarTestComparison, ComparisonFloatFirstOrder) { test_comparison<float, false>(); }
TEST(ScalarTestComparison, ComparisonDoubleFirstOrder) { test_comparison<double, false>(); }
TEST(ScalarTestComparison, ComparisonLongDoubleFirstOrder) { test_comparison<long double, false>(); }
TEST(ScalarTestComparison, ComparisonFloatSecondOrder) { test_comparison<float, true>(); }
TEST(ScalarTestComparison, ComparisonDoubleSecondOrder) { test_comparison<double, true>(); }
TEST(ScalarTestComparison, ComparisonLongDoubleSecondOrder) { test_comparison<long double, true>(); }

TEST(ScalarTestComparison, ComparisonDoubleFirstOrderDynamic) { test_comparison<double, false, true>(); }
TEST(ScalarTestComparison, ComparisonDoubleSecondOrderDynamic) { test_comparison<double, true, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_min_max()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT> a(1.0, 2.0, 3.0);
    TinyAD::Scalar<dim, PassiveT> b(2.0, 3.0, 4.0);

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

TEST(ScalarTestComparison, MinMaxFloat) { test_min_max<float>(); }
TEST(ScalarTestComparison, MinMaxDouble) { test_min_max<double>(); }
TEST(ScalarTestComparison, MinMaxLongDouble) { test_min_max<long double>(); }

TEST(ScalarTestComparison, MinMaxDoubleDynamic) { test_min_max<double, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_clamp()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT> x(4.0, 3.0, 2.0);

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

TEST(ScalarTestComparison, ClampFloat) { test_clamp<float>(); }
TEST(ScalarTestComparison, ClampDouble) { test_clamp<double>(); }
TEST(ScalarTestComparison, ClampLongDouble) { test_clamp<long double>(); }

//TEST(ScalarTestComparison, ClampDoubleDynamic) { test_clamp<double, true>(); } // clamp(TinyAD::Scalar, double) not implemented in for dynamic mode
