/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/ScalarFunction.hh>

template <typename PassiveT>
void test_1d()
{
    // 1 1D variable, 1 element
    // Convex quadratic function
    auto func = TinyAD::scalar_function<1, PassiveT>(TinyAD::range(1));
    func.template add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 1> v = element.variables(0);

        T x = v[0];
        return 2.0 * sqr(x) + x + 1.0;
    });

    using Vector1 = Eigen::Matrix<PassiveT, 1, 1>;
    auto [f, g, H] = func.eval_with_hessian_proj(Vector1(1.0));

    const PassiveT eps = 1e-16;
    ASSERT_NEAR(f, 4.0, eps);
    ASSERT_NEAR(g[0], 5.0, eps);
    ASSERT_NEAR(H.coeff(0, 0), 4.0, eps);
}

TEST(ScalarFunctionTest, 1DFloat) { test_1d<float>(); }
TEST(ScalarFunctionTest, 1DDouble) { test_1d<double>(); }
TEST(ScalarFunctionTest, 1DLongDouble) { test_1d<long double>(); }

template <typename PassiveT>
void test_2d()
{
    // 1 2D variable, 1 element
    // Convex quadratic function
    auto func = TinyAD::scalar_function<2, PassiveT>(TinyAD::range(1));
    func.template add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<T> x = element.variables(0);

        return 2.0 * sqr(x[0]) + 2.0 * x[0] * x[1] + sqr(x[1]) + x[0] + 1.0;
    });

    const Eigen::Vector2<PassiveT> x(1.0, 2.0);

    auto [f, g, H] = func.eval_with_derivatives(x);
    auto [f2, g2, H_proj] = func.eval_with_hessian_proj(x);

    const PassiveT eps = 1e-16;
    ASSERT_NEAR(f, 12, eps);
    ASSERT_NEAR(f2, 12, eps);
    ASSERT_NEAR(g[0], 9.0, eps);
    ASSERT_NEAR(g[1], 6.0, eps);
    ASSERT_NEAR(g2[0], 9.0, eps);
    ASSERT_NEAR(g2[1], 6.0, eps);
    ASSERT_NEAR(H.coeff(0, 0), 4.0, eps);
    ASSERT_NEAR(H.coeff(0, 1), 2.0, eps);
    ASSERT_NEAR(H.coeff(1, 0), 2.0, eps);
    ASSERT_NEAR(H.coeff(1, 1), 2.0, eps);
    ASSERT_NEAR(H_proj.coeff(0, 0), 4.0, eps);
    ASSERT_NEAR(H_proj.coeff(0, 1), 2.0, eps);
    ASSERT_NEAR(H_proj.coeff(1, 0), 2.0, eps);
    ASSERT_NEAR(H_proj.coeff(1, 1), 2.0, eps);
}

TEST(ScalarFunctionTest, 2DFloat) { test_2d<float>(); }
TEST(ScalarFunctionTest, 2DDouble) { test_2d<double>(); }
TEST(ScalarFunctionTest, 2DLongDouble) { test_2d<long double>(); }

template <typename PassiveT>
void test_2d_non_convex()
{
    // 1 2D variable, 1 element
    // Non-convex quadratic function
    auto func = TinyAD::scalar_function<2, PassiveT>(TinyAD::range(1));
    func.template add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<T> x = element.variables(0);

        return -(2.0 * sqr(x[0]) + 2.0 * x[0] * x[1] + sqr(x[1]) + x[0] + 1.0);
    });

    const Eigen::Vector2<PassiveT> x(1.0, 2.0);

    auto [f, g, H] = func.eval_with_derivatives(x);
    auto [f2, g2, H_proj] = func.eval_with_hessian_proj(x);

    const PassiveT eps = 1e-16;
    ASSERT_NEAR(f, -12, eps);
    ASSERT_NEAR(f2, -12, eps);
    ASSERT_NEAR(g[0], -9.0, eps);
    ASSERT_NEAR(g[1], -6.0, eps);
    ASSERT_NEAR(g2[0], -9.0, eps);
    ASSERT_NEAR(g2[1], -6.0, eps);
    ASSERT_NEAR(H.coeff(0, 0), -4.0, eps);
    ASSERT_NEAR(H.coeff(0, 1), -2.0, eps);
    ASSERT_NEAR(H.coeff(1, 0), -2.0, eps);
    ASSERT_NEAR(H.coeff(1, 1), -2.0, eps);

    // Assert positive-definite
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2<PassiveT>> eig(H_proj.toDense());
    ASSERT_GT(eig.eigenvalues()[0], 0.0);
    ASSERT_GT(eig.eigenvalues()[1], 0.0);
}

TEST(ScalarFunctionTest, 2DNonConvexFloat) { test_2d_non_convex<float>(); };
TEST(ScalarFunctionTest, 2DNonConvexDouble) { test_2d_non_convex<double>(); };
TEST(ScalarFunctionTest, 2DNonConvexLongDouble) { test_2d_non_convex<long double>(); };

TEST(ScalarFunctionTest, ElementVariables)
{
    // 2 2D variables, 1 element
    auto func = TinyAD::scalar_function<2>(TinyAD::range(2));
    func.template add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 2> v = element.variables(0);
        Eigen::Vector<T, 2> v2 = element.variables(0);
        // Eigen::Vector<T, 2> v3 = element.variables(1); // [ERROR] Too many variables requested.

        TINYAD_ASSERT_EQ(v, v2);
        if constexpr (TINYAD_ACTIVE_MODE(element))
        {
            TINYAD_ASSERT_EQ(v[0].grad, v2[0].grad);
            TINYAD_ASSERT_EQ(v[1].grad, v2[1].grad);
            TINYAD_ASSERT_EQ(v[0].Hess, v2[0].Hess);
            TINYAD_ASSERT_EQ(v[1].Hess, v2[1].Hess);
        }

        return 0;
    });

    const Eigen::Vector4d x = Eigen::Vector4d::Zero();
    func.eval_with_hessian_proj(x);
    func.eval(x);
}

TEST(ScalarFunctionTest, Move)
{
    // Test default constructor
    {
        TinyAD::ScalarFunction<2, double, Eigen::Index> empty;
        Eigen::Vector<double, 0> x;
        TINYAD_ASSERT_EQ(empty.eval(x), 0.0);
        empty.eval_with_hessian_proj(x);
    }

    // 1 2D variable, 1 element
    auto func1 = TinyAD::scalar_function<2>(TinyAD::range(1));
    func1.template add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<T> v = element.variables(0);
        return v.sum();
    });
    const Eigen::Vector2d x = Eigen::Vector2d::Constant(1.0);
    TINYAD_ASSERT_EQ(func1.eval(x), 2.0);

    // Test move constructor
    auto func2 = std::move(func1);
    TINYAD_ASSERT_EQ(func2.eval(x), 2.0);

    // Test move assignment
    TinyAD::ScalarFunction<2, double, Eigen::Index> func3;
    func3 = std::move(func2);
    TINYAD_ASSERT_EQ(func3.eval(x), 2.0);
}
