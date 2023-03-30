/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool with_hessian>
void test_constructors_static()
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

TEST(ScalarTest, ConstructorsFloatFirstOrder) { test_constructors_static<float, false>(); }
TEST(ScalarTest, ConstructorsDoubleFirstOrder) { test_constructors_static<double, false>(); }
TEST(ScalarTest, ConstructorsLongDoubleFirstOrder) { test_constructors_static<long double, false>(); }
TEST(ScalarTest, ConstructorsFloatSecondOrder) { test_constructors_static<float, true>(); }
TEST(ScalarTest, ConstructorsDoubleSecondOrder) { test_constructors_static<double, true>(); }
TEST(ScalarTest, ConstructorsLongDoubleSecondOrder) { test_constructors_static<long double, true>(); }

template <typename PassiveT, bool with_hessian>
void test_constructors_dynamic()
{
    static_assert(std::is_copy_constructible<TinyAD::Scalar<Eigen::Dynamic, PassiveT>>::value, "");
    static_assert(std::is_move_constructible<TinyAD::Scalar<Eigen::Dynamic, PassiveT>>::value, "");
    static_assert(std::is_copy_assignable<TinyAD::Scalar<Eigen::Dynamic, PassiveT>>::value, "");
    static_assert(std::is_move_assignable<TinyAD::Scalar<Eigen::Dynamic, PassiveT>>::value, "");

    const Eigen::Index k = 2;

    {
        // Active variable
        using ADouble = TinyAD::Scalar<Eigen::Dynamic, PassiveT, with_hessian>;
        ADouble a = ADouble::make_active(4.0, 0, k);
        ASSERT_EQ(a.val, 4.0);
        ASSERT_EQ(a.grad[0], 1.0);
        ASSERT_EQ(a.grad[1], 0.0);
        ASSERT_TRUE(a.Hess.isZero());

        // Passive variable
        ADouble b = ADouble::make_passive(5.0, k);
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

TEST(ScalarTest, ConstructorsFloatFirstOrderDynamic) { test_constructors_dynamic<float, false>(); }
TEST(ScalarTest, ConstructorsDoubleFirstOrderDynamic) { test_constructors_dynamic<double, false>(); }
TEST(ScalarTest, ConstructorsLongDoubleFirstOrderDynamic) { test_constructors_dynamic<long double, false>(); }
TEST(ScalarTest, ConstructorsFloatSecondOrderDynamic) { test_constructors_dynamic<float, true>(); }
TEST(ScalarTest, ConstructorsDoubleSecondOrderDynamic) { test_constructors_dynamic<double, true>(); }
TEST(ScalarTest, ConstructorsLongDoubleSecondOrderDynamic) { test_constructors_dynamic<long double, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_to_passive()
{
    {
        // make_active()
        constexpr int dim = dynamic ? Eigen::Dynamic : 2;
        const auto v = TinyAD::Scalar<dim, PassiveT, with_hessian>::make_active({ 2.0, 4.0 });
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
        constexpr int dim = dynamic ? Eigen::Dynamic : 4;
        using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
        const Eigen::Vector<ADouble, dim> v = ADouble::make_active({ 1.0, 2.0, 3.0, 4.0 });
        Eigen::Matrix<ADouble, 2, 2> M;
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

TEST(ScalarTest, ToPassiveFloatFirstOrder) { test_to_passive<float, false>(); }
TEST(ScalarTest, ToPassiveDoubleFirstOrder) { test_to_passive<double, false>(); }
TEST(ScalarTest, ToPassiveLongDoubleFirstOrder) { test_to_passive<long double, false>(); }
TEST(ScalarTest, ToPassiveFloatSecondOrder) { test_to_passive<float, true>(); }
TEST(ScalarTest, ToPassiveDoubleSecondOrder) { test_to_passive<double, true>(); }
TEST(ScalarTest, ToPassiveLongDoubleSecondOrder) { test_to_passive<long double, true>(); }

TEST(ScalarTest, ToPassiveFirstOrderDynamic) { test_to_passive<double, false, true>(); }
TEST(ScalarTest, ToPassiveSecondOrderDynamic) { test_to_passive<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_quadratic()
{
    // f(a) = a^2 + a + 2 at a=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::make_active(1.0, 0, 1);
    const ADouble f = sqr(a) + a + 2.0;
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

TEST(ScalarTest, QuadraticDoubleFirstOrderDynamic) { test_quadratic<double, false, true>(); }
TEST(ScalarTest, QuadraticDoubleSecondOrderDynamic) { test_quadratic<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_unary_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = -a;
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

TEST(ScalarTest, UnaryMinusDoubleFirstOrderDynamic) { test_unary_minus<double, false, true>(); }
TEST(ScalarTest, UnaryMinusDoubleSecondOrderDynamic) { test_unary_minus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sqrt()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = sqrt(a);
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

TEST(ScalarTest, SqrtDoubleFirstOrderDynamic) { test_sqrt<double, false, true>(); }
TEST(ScalarTest, SqrtDoubleSecondOrderDynamic) { test_sqrt<double, true, true>(); }

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

TEST(ScalarTest, SqrFloatFirstOrder) { test_sqr<float, false>(); }
TEST(ScalarTest, SqrDoubleFirstOrder) { test_sqr<double, false>(); }
TEST(ScalarTest, SqrLongDoubleFirstOrder) { test_sqr<long double, false>(); }
TEST(ScalarTest, SqrFloatSecondOrder) { test_sqr<float, true>(); }
TEST(ScalarTest, SqrDoubleSecondOrder) { test_sqr<double, true>(); }
TEST(ScalarTest, SqrLongDoubleSecondOrder) { test_sqr<long double, true>(); }

TEST(ScalarTest, SqrDoubleFirstOrderDynamic) { test_sqr<double, false, true>(); }
TEST(ScalarTest, SqrDoubleSecondOrderDynamic) { test_sqr<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_pow_int()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = pow(a, 3);
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

TEST(ScalarTest, PowDoubleFirstOrderDynamic) { test_pow_int<double, false, true>(); }
TEST(ScalarTest, PowDoubleSecondOrderDynamic) { test_pow_int<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_pow_real()
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = pow(a, PassiveT(3.0 / 2.0));
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

TEST(ScalarTest, PowDoubleRealFirstOrderDynamic) { test_pow_real<double, false, true>(); }
TEST(ScalarTest, PowDoubleRealSecondOrderDynamic) { test_pow_real<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_fabs()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;

    {   // a(x) = x^3 at x = 1
        const TinyAD::Scalar<dim, PassiveT, with_hessian> a(1.0, 3.0, 6.0);
        const TinyAD::Scalar<dim, PassiveT, with_hessian> f = fabs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        const TinyAD::Scalar<dim, PassiveT, with_hessian> a(-1.0, 3.0, -6.0);
        const TinyAD::Scalar<dim, PassiveT, with_hessian> f = fabs(a);
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

TEST(ScalarTest, FabsDoubleFirstOrderDynamic) { test_fabs<double, false, true>(); }
TEST(ScalarTest, FabsDoubleSecondOrderDynamic) { test_fabs<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_abs()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;

    {   // a(x) = x^3 at x = 1
        const TinyAD::Scalar<dim, PassiveT, with_hessian> a(1.0, 3.0, 6.0);
        const TinyAD::Scalar<dim, PassiveT, with_hessian> f = abs(a);
        ASSERT_NEAR(f.val, 1.0, 1e-12);
        ASSERT_NEAR(f.grad(0), 3.0, 1e-12);
        if constexpr (with_hessian)
        {
            ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
            TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
        }
    }

    {   // a(x) = x^3 at x = -1
        const TinyAD::Scalar<dim, PassiveT, with_hessian> a(-1.0, 3.0, -6.0);
        const TinyAD::Scalar<dim, PassiveT, with_hessian> f = abs(a);
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

TEST(ScalarTest, AbsDoubleFirstOrderDynamic) { test_abs<double, false, true>(); }
TEST(ScalarTest, AbsDoubleSecondOrderDynamic) { test_abs<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_exp(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = exp(a);
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

TEST(ScalarTest, ExpDoubleFirstOrderDynamic) { test_exp<double, false, true>(1e-12); }
TEST(ScalarTest, ExpDoubleSecondOrderDynamic) { test_exp<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = log(a);
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

TEST(ScalarTest, LogDoubleFirstOrderDynamic) { test_log<double, false, true>(1e-12); }
TEST(ScalarTest, LogDoubleSecondOrderDynamic) { test_log<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log2(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = log2(a);
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

TEST(ScalarTest, Log2DoubleFirstOrderDynamic) { test_log2<double, false, true>(1e-12); }
TEST(ScalarTest, Log2DoubleSecondOrderDynamic) { test_log2<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_log10(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = log10(a);
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

TEST(ScalarTest, Log10DoubleFirstOrderDynamic) { test_log10<double, false, true>(1e-12); }
TEST(ScalarTest, Log10DoubleSecondOrderDynamic) { test_log10<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sin(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = sin(a);
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

TEST(ScalarTest, SinDoubleFirstOrderDynamic) { test_sin<double, false, true>(1e-12); }
TEST(ScalarTest, SinDoubleSecondOrderDynamic) { test_sin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_cos(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = cos(a);
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

TEST(ScalarTest, CosDoubleFirstOrderDynamic) { test_cos<double, false, true>(1e-12); }
TEST(ScalarTest, CosDoubleSecondOrderDynamic) { test_cos<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_tan(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = tan(a);
    ASSERT_NEAR(f.val, std::tan(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cos(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 4.0 * (1.0 + 9.0 * std::tan(4.0)) / (1.0 + std::cos(8.0)), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, TanFloatFirstOrder) { test_tan<float, false>(1e-4f); }
TEST(ScalarTest, TanDoubleFirstOrder) { test_tan<double, false>(1e-12); }
TEST(ScalarTest, TanLongDoubleFirstOrder) { test_tan<long double, false>(1e-12); }
TEST(ScalarTest, TanFloatSecondOrder) { test_tan<float, true>(1e-4f); }
TEST(ScalarTest, TanDoubleSecondOrder) { test_tan<double, true>(1e-12); }
TEST(ScalarTest, TanLongDoubleSecondOrder) { test_tan<long double, true>(1e-12); }

TEST(ScalarTest, TanDoubleFirstOrderDynamic) { test_tan<double, false, true>(1e-12); }
TEST(ScalarTest, TanDoubleSecondOrderDynamic) { test_tan<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_asin(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = asin(a);
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

TEST(ScalarTest, ASinDoubleFirstOrderDynamic) { test_asin<double, false, true>(1e-12); }
TEST(ScalarTest, ASinDoubleSecondOrderDynamic) { test_asin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_acos(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = acos(a);
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

TEST(ScalarTest, ACosDoubleFirstOrderDynamic) { test_asin<double, false, true>(1e-12); }
TEST(ScalarTest, ACosDoubleSecondOrderDynamic) { test_asin<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atan(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = atan(a);
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

TEST(ScalarTest, ATanDoubleFirstOrderDynamic) { test_atan<double, false, true>(1e-12); }
TEST(ScalarTest, ATanDoubleSecondOrderDynamic) { test_atan<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sinh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = sinh(a);
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

TEST(ScalarTest, SinhDoubleFirstOrderDynamic) { test_sinh<double, false, true>(1e-12); }
TEST(ScalarTest, SinhDoubleSecondOrderDynamic) { test_sinh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_cosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = cosh(a);
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

TEST(ScalarTest, CoshDoubleFirstOrderDynamic) { test_cosh<double, false, true>(1e-12); }
TEST(ScalarTest, CoshDoubleSecondOrderDynamic) { test_cosh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_tanh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = tanh(a);
    ASSERT_NEAR(f.val, std::tanh(4.0), _eps);
    ASSERT_NEAR(f.grad(0), 3.0 / sqr(std::cosh(4.0)), _eps);
    if constexpr (with_hessian)
    {
        ASSERT_NEAR(f.Hess(0, 0), 2.0 * (1.0 - 9.0 * std::sinh(4.0) / std::cosh(4.0)) / (sqr(std::cosh(4.0))), _eps);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, _eps);
    }
}

TEST(ScalarTest, TanhFloatFirstOrder) { test_tanh<float, false>(1e-4f); }
TEST(ScalarTest, TanhDoubleFirstOrder) { test_tanh<double, false>(1e-12); }
TEST(ScalarTest, TanhLongDoubleFirstOrder) { test_tanh<long double, false>(1e-12); }
TEST(ScalarTest, TanhFloatSecondOrder) { test_tanh<float, true>(1e-4f); }
TEST(ScalarTest, TanhDoubleSecondOrder) { test_tanh<double, true>(1e-12); }
TEST(ScalarTest, TanhLongDoubleSecondOrder) { test_tanh<long double, true>(1e-12); }

TEST(ScalarTest, TanhDoubleFirstOrderDynamic) { test_tanh<double, false, true>(1e-12); }
TEST(ScalarTest, TanhDoubleSecondOrderDynamic) { test_tanh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_asinh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = asinh(a);
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

TEST(ScalarTest, ASinhDoubleFirstOrderDynamic) { test_asinh<double, false, true>(1e-12); }
TEST(ScalarTest, ASinhDoubleSecondOrderDynamic) { test_asinh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_acosh(const PassiveT _eps)
{
    // a(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = acosh(a);
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

TEST(ScalarTest, ACoshDoubleFirstOrderDynamic) { test_acosh<double, false, true>(1e-12); }
TEST(ScalarTest, ACoshDoubleSecondOrderDynamic) { test_acosh<double, true, true>(1e-12); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_atanh(const PassiveT _eps)
{
    // a(x) = x^2 + x - 1.5 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    const TinyAD::Scalar<dim, PassiveT, with_hessian> a(0.5, 3.0, 2.0);
    const TinyAD::Scalar<dim, PassiveT, with_hessian> f = atanh(a);
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

TEST(ScalarTest, ATanhDoubleFirstOrderDynamic) { test_atanh<double, false, true>(1e-12); }
TEST(ScalarTest, ATanhDoubleSecondOrderDynamic) { test_atanh<double, true, true>(1e-12); }

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

TEST(ScalarTest, IsnanIsinfFloat) { test_isnan_isinf<float>(); }
TEST(ScalarTest, IsnanIsinfDouble) { test_isnan_isinf<double>(); }
TEST(ScalarTest, IsnanIsinfLongDouble) { test_isnan_isinf<long double>(); }

TEST(ScalarTest, IsnanIsinfDoubleDynamic) { test_isnan_isinf<double, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_plus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

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

TEST(ScalarTest, PlusDoubleFirstOrderDynamic) { test_plus<double, false, true>(); }
TEST(ScalarTest, PlusDoubleSecondOrderDynamic) { test_plus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_minus()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

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

TEST(ScalarTest, MinusDoubleFirstOrderDynamic) { test_minus<double, false, true>(); }
TEST(ScalarTest, MinusDoubleSecondOrderDynamic) { test_minus<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_mult()
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT, with_hessian> a(4.0, 3.0, 2.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> b(0.0, 1.0, 4.0);

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

TEST(ScalarTest, MultDoubleFirstOrderDynamic) { test_mult<double, false, true>(); }
TEST(ScalarTest, MultDoubleSecondOrderDynamic) { test_mult<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_div()
{
    // a(x) = x^3 - x^2 + 1 at x=1
    // b(x) = x^2 + x + 2 at x=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    TinyAD::Scalar<dim, PassiveT, with_hessian> a(1.0, 1.0, 4.0);
    TinyAD::Scalar<dim, PassiveT, with_hessian> b(4.0, 3.0, 2.0);

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

TEST(ScalarTest, DivDoubleFirstOrderDynamic) { test_div<double, false, true>(); }
TEST(ScalarTest, DivDoubleSecondOrderDynamic) { test_div<double, true, true>(); }

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

TEST(ScalarTest, Atan2ConstFloatFirstOrder) { test_atan2_const<float, false>(1e-4f); }
TEST(ScalarTest, Atan2ConstDoubleFirstOrder) { test_atan2_const<double, false>(1e-12); }
TEST(ScalarTest, Atan2ConstLongDoubleFirstOrder) { test_atan2_const<long double, false>(1e-12); }
TEST(ScalarTest, Atan2ConstFloatSecondOrder) { test_atan2_const<float, true>(1e-4f); }
TEST(ScalarTest, Atan2ConstDoubleSecondOrder) { test_atan2_const<double, true>(1e-12); }
TEST(ScalarTest, Atan2ConstLongDoubleSecondOrder) { test_atan2_const<long double, true>(1e-12); }

TEST(ScalarTest, Atan2ConstDoubleFirstOrderDynamic) { test_atan2_const<double, false, true>(1e-12); }
TEST(ScalarTest, Atan2ConstDoubleSecondOrderDynamic) { test_atan2_const<double, true, true>(1e-12); }

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

TEST(ScalarTest, Atan2_1FloatFirstOrder) { test_atan2_1<float, false>(1e-4f); }
TEST(ScalarTest, Atan2_1DoubleFirstOrder) { test_atan2_1<double, false>(1e-12); }
TEST(ScalarTest, Atan2_1LongDoubleFirstOrder) { test_atan2_1<long double, false>(1e-12); }
TEST(ScalarTest, Atan2_1FloatSecondOrder) { test_atan2_1<float, true>(1e-4f); }
TEST(ScalarTest, Atan2_1DoubleSecondOrder) { test_atan2_1<double, true>(1e-12); }
TEST(ScalarTest, Atan2_1LongDoubleSecondOrder) { test_atan2_1<long double, true>(1e-12); }

TEST(ScalarTest, Atan2_1DoubleFirstOrderDynamic) { test_atan2_1<double, false, true>(1e-12); }
TEST(ScalarTest, Atan2_1DoubleSecondOrderDynamic) { test_atan2_1<double, true, true>(1e-12); }

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

TEST(ScalarTest, Atan2_2FloatFirstOrder) { test_atan2_2<float, false>(1e-4f); }
TEST(ScalarTest, Atan2_2DoubleFirstOrder) { test_atan2_2<double, false>(1e-12); }
TEST(ScalarTest, Atan2_2LongDoubleFirstOrder) { test_atan2_2<long double, false>(1e-12); }
TEST(ScalarTest, Atan2_2FloatSecondOrder) { test_atan2_2<float, true>(1e-4f); }
TEST(ScalarTest, Atan2_2DoubleSecondOrder) { test_atan2_2<double, true>(1e-12); }
TEST(ScalarTest, Atan2_2LongDoubleSecondOrder) { test_atan2_2<long double, true>(1e-12); }

TEST(ScalarTest, Atan2_2ConstDoubleFirstOrderDynamic) { test_atan2_2<double, false, true>(1e-12); }
TEST(ScalarTest, Atan2_2ConstDoubleSecondOrderDynamic) { test_atan2_2<double, true, true>(1e-12); }

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

TEST(ScalarTest, HypotFloatFirstOrder) { test_hypot<float, false>(1e-7f); }
TEST(ScalarTest, HypotDoubleFirstOrder) { test_hypot<double, false>(1e-12); }
TEST(ScalarTest, HypotLongDoubleFirstOrder) { test_hypot<long double, false>(1e-14); }
TEST(ScalarTest, HypotFloatSecondOrder) { test_hypot<float, true>(1e-7f); }
TEST(ScalarTest, HypotDoubleSecondOrder) { test_hypot<double, true>(1e-12); }
TEST(ScalarTest, HypotLongDoubleSecondOrder) { test_hypot<long double, true>(1e-14); }

TEST(ScalarTest, HypotDoubleFirstOrderDynamic) { test_hypot<double, false, true>(1e-12); }
TEST(ScalarTest, HypotDoubleSecondOrderDynamic) { test_hypot<double, true, true>(1e-12); }

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

TEST(ScalarTest, Div2dFloatFirstOrder) { test_div2d<float, false>(1e-4f); }
TEST(ScalarTest, Div2dDoubleFirstOrder) { test_div2d<double, false>(1e-12); }
TEST(ScalarTest, Div2dLongDoubleFirstOrder) { test_div2d<long double, false>(1e-12); }
TEST(ScalarTest, Div2dFloatSecondOrder) { test_div2d<float, true>(1e-4f); }
TEST(ScalarTest, Div2dDoubleSecondOrder) { test_div2d<double, true>(1e-12); }
TEST(ScalarTest, Div2dLongDoubleSecondOrder) { test_div2d<long double, true>(1e-12); }

TEST(ScalarTest, Div2dDoubleFirstOrderDynamic) { test_div2d<double, false, true>(1e-12); }
TEST(ScalarTest, Div2dDoubleSecondOrderDynamic) { test_div2d<double, true, true>(1e-12); }

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

TEST(ScalarTest, Div2d_2FloatFirstOrder) { test_div2d_2<float, false>(1e-4f); }
TEST(ScalarTest, Div2d_2DoubleFirstOrder) { test_div2d_2<double, false>(1e-12); }
TEST(ScalarTest, Div2d_2LongDoubleFirstOrder) { test_div2d_2<long double, false>(1e-12); }
TEST(ScalarTest, Div2d_2FloatSecondOrder) { test_div2d_2<float, true>(1e-4f); }
TEST(ScalarTest, Div2d_2DoubleSecondOrder) { test_div2d_2<double, true>(1e-12); }
TEST(ScalarTest, Div2d_2LongDoubleSecondOrder) { test_div2d_2<long double, true>(1e-12); }

TEST(ScalarTest, Div2d_2DoubleFirstOrderDynamic) { test_div2d_2<double, false, true>(1e-12); }
TEST(ScalarTest, Div2d_2DoubleSecondOrderDynamic) { test_div2d_2<double, true, true>(1e-12); }

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

TEST(ScalarTest, PlusMinusMultDiv2dFloatFirstOrder) { test_plus_minus_mult_div_2d<float, false>(1e-4f); }
TEST(ScalarTest, PlusMinusMultDiv2dDoubleFirstOrder) { test_plus_minus_mult_div_2d<double, false>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dLongDoubleFirstOrder) { test_plus_minus_mult_div_2d<long double, false>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dFloatSecondOrder) { test_plus_minus_mult_div_2d<float, true>(1e-4f); }
TEST(ScalarTest, PlusMinusMultDiv2dDoubleSecondOrder) { test_plus_minus_mult_div_2d<double, true>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dLongDoubleSecondOrder) { test_plus_minus_mult_div_2d<long double, true>(1e-12); }

TEST(ScalarTest, PlusMinusMultDiv2dDoubleFirstOrderDynamic) { test_plus_minus_mult_div_2d<double, false, true>(1e-12); }
TEST(ScalarTest, PlusMinusMultDiv2dDoubleSecondOrderDynamic) { test_plus_minus_mult_div_2d<double, true, true>(1e-12); }

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

TEST(ScalarTest, ComparisonFloatFirstOrder) { test_comparison<float, false>(); }
TEST(ScalarTest, ComparisonDoubleFirstOrder) { test_comparison<double, false>(); }
TEST(ScalarTest, ComparisonLongDoubleFirstOrder) { test_comparison<long double, false>(); }
TEST(ScalarTest, ComparisonFloatSecondOrder) { test_comparison<float, true>(); }
TEST(ScalarTest, ComparisonDoubleSecondOrder) { test_comparison<double, true>(); }
TEST(ScalarTest, ComparisonLongDoubleSecondOrder) { test_comparison<long double, true>(); }

TEST(ScalarTest, ComparisonDoubleFirstOrderDynamic) { test_comparison<double, false, true>(); }
TEST(ScalarTest, ComparisonDoubleSecondOrderDynamic) { test_comparison<double, true, true>(); }

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

TEST(ScalarTest, MinMaxFloat) { test_min_max<float>(); }
TEST(ScalarTest, MinMaxDouble) { test_min_max<double>(); }
TEST(ScalarTest, MinMaxLongDouble) { test_min_max<long double>(); }

TEST(ScalarTest, MinMaxDoubleDynamic) { test_min_max<double, true>(); }

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

TEST(ScalarTest, ClampFloat) { test_clamp<float>(); }
TEST(ScalarTest, ClampDouble) { test_clamp<double>(); }
TEST(ScalarTest, ClampLongDouble) { test_clamp<long double>(); }

//TEST(ScalarTest, ClampDoubleDynamic) { test_clamp<double, true>(); } // clamp(TinyAD::Scalar, double) not implemented in for dynamic mode

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sphere()
{
    // f: R^2 -> R^3
    // f(phi, psi) = (sin(phi) * cos(psi), sin(phi) * sin(psi), cos(phi))
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble alpha = ADouble::make_active((PassiveT)M_PI / 8.0, 0, 2);
    ADouble beta = ADouble::make_active((PassiveT)M_PI / 8.0, 1, 2);
    const auto f = Eigen::Matrix<ADouble, 3, 1>(
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

TEST(ScalarTest, SphereDoubleFirstOrderDynamic) { test_sphere<double, false, true>(); }
TEST(ScalarTest, SphereDoubleSecondOrderDynamic) { test_sphere<double, true, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_min_quadric()
{
    // Variable vector in R^3
    constexpr int dim = dynamic ? Eigen::Dynamic : 3;
    using ADouble = TinyAD::Scalar<dim, PassiveT>;
    const Eigen::Vector<ADouble, dim> x = ADouble::make_active({ 0.0, 0.0, 0.0 });

    // Quadratic function
    const ADouble f = sqr(x[0]) + 2.0 * sqr(x[1]) + 6.0 * sqr(x[2]) + x[0] - 2.0 * x[1] + 6.0 * x[2] + 10;

    // Solve for minimum
    const Eigen::Vector<PassiveT, dim> x_min = -f.Hess.inverse() * f.grad;

    ASSERT_NEAR(x_min.x(), -0.5, 1e-12);
    ASSERT_NEAR(x_min.y(), 0.5, 1e-12);
    ASSERT_NEAR(x_min.z(), -0.5, 1e-12);
}

TEST(ScalarTest, MinQuadraticFloat) { test_min_quadric<float>(); }
TEST(ScalarTest, MinQuadraticDouble) { test_min_quadric<double>(); }
TEST(ScalarTest, MinQuadraticLongDouble) { test_min_quadric<long double>(); }

TEST(ScalarTest, MinQuadraticDoubleDynamic) { test_min_quadric<double, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_triangle_distortion()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 6;
    using ADouble = TinyAD::Scalar<dim, PassiveT>;

    // Passive rest-state triangle ar, br, cr
    const Eigen::Matrix<PassiveT, 2, 1> ar(1.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> br(2.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> cr(1.0, 2.0);
    const Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);

    // Active variable vector for vertex positions a, b, c
    const Eigen::Vector<ADouble, dim> x = ADouble::make_active({
        10.0, 1.0,
        15.0, 3.0,
        2.0, 2.0,
    });
    const Eigen::Matrix<ADouble, 2, 1> a(x[0], x[1]);
    const Eigen::Matrix<ADouble, 2, 1> b(x[2], x[3]);
    const Eigen::Matrix<ADouble, 2, 1> c(x[4], x[5]);
    const Eigen::Matrix<ADouble, 2, 2> M = TinyAD::col_mat(b - a, c - a);

    const Eigen::Matrix<ADouble, 2, 2> J = M * Mr.inverse();
    const ADouble E = J.squaredNorm() + J.inverse().squaredNorm();
    TINYAD_ASSERT_FINITE(E.val);
    TINYAD_ASSERT_FINITE_MAT(E.grad);
    TINYAD_ASSERT_FINITE_MAT(E.Hess);
}

TEST(ScalarTest, TriangleDistortionFloat) { test_triangle_distortion<float>(); }
TEST(ScalarTest, TriangleDistortionDouble) { test_triangle_distortion<double>(); }
TEST(ScalarTest, TriangleDistortionLongDouble) { test_triangle_distortion<long double>(); }

//TEST(ScalarTest, TriangleDistortionDoubleDynamic) { test_triangle_distortion<double, true>(); } // Not available, b/c squaredNorm() needs default constructor "Scalar(0)" to be implemented, which is currently not the case in dynamic mode.
