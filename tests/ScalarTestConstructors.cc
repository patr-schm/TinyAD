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

TEST(ScalarTestConstructors, ConstructorsFloatFirstOrder) { test_constructors_static<float, false>(); }
TEST(ScalarTestConstructors, ConstructorsDoubleFirstOrder) { test_constructors_static<double, false>(); }
TEST(ScalarTestConstructors, ConstructorsLongDoubleFirstOrder) { test_constructors_static<long double, false>(); }
TEST(ScalarTestConstructors, ConstructorsFloatSecondOrder) { test_constructors_static<float, true>(); }
TEST(ScalarTestConstructors, ConstructorsDoubleSecondOrder) { test_constructors_static<double, true>(); }
TEST(ScalarTestConstructors, ConstructorsLongDoubleSecondOrder) { test_constructors_static<long double, true>(); }

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

TEST(ScalarTestConstructors, ConstructorsFloatFirstOrderDynamic) { test_constructors_dynamic<float, false>(); }
TEST(ScalarTestConstructors, ConstructorsDoubleFirstOrderDynamic) { test_constructors_dynamic<double, false>(); }
TEST(ScalarTestConstructors, ConstructorsLongDoubleFirstOrderDynamic) { test_constructors_dynamic<long double, false>(); }
TEST(ScalarTestConstructors, ConstructorsFloatSecondOrderDynamic) { test_constructors_dynamic<float, true>(); }
TEST(ScalarTestConstructors, ConstructorsDoubleSecondOrderDynamic) { test_constructors_dynamic<double, true>(); }
TEST(ScalarTestConstructors, ConstructorsLongDoubleSecondOrderDynamic) { test_constructors_dynamic<long double, true>(); }

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

TEST(ScalarTestConstructors, ToPassiveFloatFirstOrder) { test_to_passive<float, false>(); }
TEST(ScalarTestConstructors, ToPassiveDoubleFirstOrder) { test_to_passive<double, false>(); }
TEST(ScalarTestConstructors, ToPassiveLongDoubleFirstOrder) { test_to_passive<long double, false>(); }
TEST(ScalarTestConstructors, ToPassiveFloatSecondOrder) { test_to_passive<float, true>(); }
TEST(ScalarTestConstructors, ToPassiveDoubleSecondOrder) { test_to_passive<double, true>(); }
TEST(ScalarTestConstructors, ToPassiveLongDoubleSecondOrder) { test_to_passive<long double, true>(); }

TEST(ScalarTestConstructors, ToPassiveFirstOrderDynamic) { test_to_passive<double, false, true>(); }
TEST(ScalarTestConstructors, ToPassiveSecondOrderDynamic) { test_to_passive<double, true, true>(); }
