/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/ScalarFunction.hh>
#include "Meshes.hh"

template <typename PassiveT>
Eigen::VectorX<PassiveT> test_2d_deformation_newton(
            const PassiveT& _eps)
{
    Eigen::MatrixX<PassiveT> V_rest;
    Eigen::MatrixX<PassiveT> V;
    Eigen::MatrixXi F;
    std::vector<Eigen::Index> b;
    std::vector<Eigen::Vector2<PassiveT>> bc;
    planar_test_mesh(V_rest, V, F, b, bc);

    // 6 2D variables
    auto func = TinyAD::scalar_function<2, PassiveT>(TinyAD::range(V.rows()));

    // Add symmetric Dirichlet energy term.
    // 4 elements using 3 variable handles each.
    func.template add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<PassiveT, 2> ar = V_rest.row(F(element.handle, 0));
        Eigen::Vector<PassiveT, 2> br = V_rest.row(F(element.handle, 1));
        Eigen::Vector<PassiveT, 2> cr = V_rest.row(F(element.handle, 2));
        Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);
        Eigen::Vector<T, 2> a = element.variables(F(element.handle, 0));
        Eigen::Vector<T, 2> b = element.variables(F(element.handle, 1));
        Eigen::Vector<T, 2> c = element.variables(F(element.handle, 2));
        Eigen::Matrix<T, 2, 2> M = TinyAD::col_mat(b - a, c - a);

        if (M.determinant() <= 0.0)
            return INFINITY;

        return ((M * Mr.inverse()).squaredNorm() + (Mr * M.inverse()).squaredNorm()) / (PassiveT)F.rows();
    });

    // Add positional penalty terms.
    // 2 elements using 1 variable handle each.
    func.template add_elements<1>(TinyAD::range(b.size()), [&] (auto& element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<PassiveT> p_target = bc[element.handle];
        Eigen::Vector2<T> p = element.variables(b[element.handle]);

        return (p_target - p).squaredNorm();
    });

    // Assemble initial x vector
    Eigen::VectorX<PassiveT> x = func.x_from_data([&] (Eigen::Index v_idx) {
        return V.row(v_idx);
    });

    // Assert number of non-zeros in Hessian
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        TINYAD_ASSERT_EQ(H_proj.nonZeros(), 4 * V.rows() + 8 * (V.rows() + F.rows() - 1));
    }

    // Optimize
    TinyAD::LinearSolver<PassiveT> solver;
    for (int i = 0; i < 10; ++i)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        Eigen::VectorX<PassiveT> d = newton_direction(g, H_proj, solver);
        x = line_search(x, d, f, g, func);
    }

    // Write final x vector to V
    func.x_to_data(x, [&] (Eigen::Index v_idx, const Eigen::Vector2<PassiveT>& p) {
        V.row(v_idx) = p;
    });

    // Assert distortion minimum has been reached and gradient is zero
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        TINYAD_ASSERT_EPS(func.eval(x), f, _eps);
        TINYAD_ASSERT_EPS(f, 4.0, _eps);
        TINYAD_ASSERT_EPS(g.cwiseAbs().maxCoeff(), 0.0, _eps);
    }

    return x;
}

TEST(NewtonTest, 2DDeformationFloat) { test_2d_deformation_newton<float>(1e-6f); }
TEST(NewtonTest, 2DDeformationDouble) { test_2d_deformation_newton<double>(1e-15); }
TEST(NewtonTest, 2DDeformationLongDouble) { test_2d_deformation_newton<long double>(1e-15); }

TEST(NewtonTest, Deterministic)
{
    std::vector<Eigen::VectorXd> xs(4);

    #pragma omp parallel for
    for (int i = 0; i < (int)xs.size(); ++i)
    {
        xs[i] = test_2d_deformation_newton<double>(1e-15);
    }

    for (int i = 0; i < (int)xs.size(); ++i)
    {
        TINYAD_ASSERT_EQ(xs[i], xs[0]);
    }
}
