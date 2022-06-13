/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/VectorFunction.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/GaussNewtonDirection.hh>
#include "Meshes.hh"

template <typename PassiveT>
Eigen::VectorX<PassiveT> test_2d_deformation_gauss_newton(
            const PassiveT& _eps)
{
    Eigen::MatrixX<PassiveT> V_rest;
    Eigen::MatrixX<PassiveT> V;
    Eigen::MatrixXi F;
    std::vector<Eigen::Index> b;
    std::vector<Eigen::Vector2<PassiveT>> bc;
    planar_test_mesh(V_rest, V, F, b, bc);

    // 6 2D variables
    // Distortion:
    //      4 triangle elements.
    //      Each triangle element uses 3 vertices.
    //      Each triangle element creates 8 residuals (one per Jacobian (or inverse Jacobian) entry).
    // Position penalty:
    //      2 vertex elements.
    //      Each vertex element uses 1 vertex.
    //      Each vertex element creates 2 residuals (one per coordinate)
    auto func_sos = TinyAD::vector_function<2, PassiveT>(TinyAD::range(V.rows()));
    func_sos.template add_elements<3, 8>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;
        Eigen::Vector<PassiveT, 2> ar = V_rest.row(F(f_idx, 0));
        Eigen::Vector<PassiveT, 2> br = V_rest.row(F(f_idx, 1));
        Eigen::Vector<PassiveT, 2> cr = V_rest.row(F(f_idx, 2));
        Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);
        Eigen::Vector<T, 2> a = element.variables(F(f_idx, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f_idx, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f_idx, 2));
        Eigen::Matrix<T, 2, 2> M = TinyAD::col_mat(b - a, c - a);

        if (M.determinant() <= 0.0)
            return Eigen::Vector<T, 8>::Constant(INFINITY);

        Eigen::Matrix2<T> J = M * Mr.inverse();
        Eigen::Matrix2<T> J_inv = Mr * M.inverse();

        Eigen::Vector<T, 8> E;
        E(0) = J(0, 0);
        E(1) = J(0, 1);
        E(2) = J(1, 0);
        E(3) = J(1, 1);
        E(4) = J_inv(0, 0);
        E(5) = J_inv(0, 1);
        E(6) = J_inv(1, 0);
        E(7) = J_inv(1, 1);

        return 1.0 / sqrt(F.rows()) * E;
    });
    func_sos.template add_elements<1, 2>(TinyAD::range(b.size()), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<PassiveT> p_target = bc[element.handle];
        Eigen::Vector2<T> p = element.variables(b[element.handle]);

        return p_target - p;
    });

    // For comparison: Standard formulation
    // 6 2D variables.
    // Element per triangle.
    // Positional penalty terms: 2 elements using 1 variable handle each.
    auto func_ref = TinyAD::scalar_function<2, PassiveT>(TinyAD::range(V.rows()));
    func_ref.template add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;
        Eigen::Vector<PassiveT, 2> ar = V_rest.row(F(f_idx, 0));
        Eigen::Vector<PassiveT, 2> br = V_rest.row(F(f_idx, 1));
        Eigen::Vector<PassiveT, 2> cr = V_rest.row(F(f_idx, 2));
        Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);
        Eigen::Vector<T, 2> a = element.variables(F(f_idx, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f_idx, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f_idx, 2));
        Eigen::Matrix<T, 2, 2> M = TinyAD::col_mat(b - a, c - a);

        if (M.determinant() <= 0.0)
            return INFINITY;

        return ((M * Mr.inverse()).squaredNorm() + (Mr * M.inverse()).squaredNorm()) / F.rows();
    });
    func_ref.template add_elements<1>(TinyAD::range(b.size()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector2<PassiveT> p_target = bc[element.handle];
        Eigen::Vector2<T> p = element.variables(b[element.handle]);

        return (p_target - p).squaredNorm();
    });

    // Assemble initial x vector
    Eigen::VectorX<PassiveT> x = func_sos.x_from_data([&] (int v_idx) {
        return V.row(v_idx);
    });

    // Optimize
    TinyAD::LinearSolver<PassiveT> solver;
    for (int i = 0; i < 20; ++i)
    {
        // Compare different ways of evaluation to reference
        auto [f_ref, g_ref, H_proj_ref] = func_ref.eval_with_hessian_proj(x);

        { // Reference function, eval()
            PassiveT f = func_ref.eval(x);
            TINYAD_ASSERT_EPS(f, f_ref, _eps);
        }

        { // Reference function, eval_with_gradient()
            auto [f, g] = func_ref.eval_with_gradient(x);
            TINYAD_ASSERT_EPS(f, f_ref, _eps);
            TINYAD_ASSERT_L((g - g_ref).cwiseAbs().maxCoeff(), _eps);
        }

        { // Sum-of-squares function, eval_sum_of_squares()
            PassiveT f = func_sos.eval_sum_of_squares(x);
            TINYAD_ASSERT_EPS(f, f_ref, _eps);
        }

        { // Sum-of-squares function, eval_sum_of_squares_with_derivatives()
            auto [f, g, r, J] = func_sos.eval_sum_of_squares_with_derivatives(x);
            TINYAD_ASSERT_EPS(f, f_ref, _eps);
            TINYAD_ASSERT_L((g - g_ref).cwiseAbs().maxCoeff(), _eps);
        }

        // Compute Gauss-Newton step
        auto [f, g, r, J] = func_sos.eval_sum_of_squares_with_derivatives(x);
        const Eigen::VectorX<PassiveT> d = TinyAD::gauss_newton_direction(r, J, solver, (PassiveT)1e-12);
        x = TinyAD::line_search(x, d, f, g, [&] (const Eigen::VectorX<PassiveT>& _x)
        {
            return func_sos.eval_sum_of_squares(_x);
        });
    }

    // Write final x vector to V
    func_sos.x_to_data(x, [&] (int v_idx, const Eigen::Vector2<PassiveT>& p) {
        V.row(v_idx) = p;
    });

    // Assert distortion minimum has been reached and gradient is zero
    {
        auto [f, g, r, J] = func_sos.eval_sum_of_squares_with_derivatives(x);
        TINYAD_ASSERT_EPS(f, 4.0, 0.1);
        TINYAD_ASSERT_EPS(g.cwiseAbs().maxCoeff(), 0.0, 0.1);
    }

    return x;
}

TEST(GaussNewtonTest, 2DDeformationGaussNewtonFloat) { test_2d_deformation_gauss_newton<float>(1e-5f); }
TEST(GaussNewtonTest, 2DDeformationGaussNewtonDouble) { test_2d_deformation_gauss_newton<double>(1e-12); }
TEST(GaussNewtonTest, 2DDeformationGaussNewtonLongDouble) { test_2d_deformation_gauss_newton<long double>(1e-14); }
