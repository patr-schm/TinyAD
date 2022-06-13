/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/Timer.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/ScalarFunction.hh>

namespace
{

/// Triangle mesh in the plane with 6 vertices and 4 faces.
template <typename PassiveT>
void planar_test_mesh(
            Eigen::MatrixX<PassiveT>& _V, Eigen::MatrixXi& _F)
{
    _V = Eigen::MatrixX<PassiveT> (6, 2);
    _V << 0.0, 0.0,
          1.0, 0.0,
          0.0, 1.0,
          1.0, 1.0,
          0.0, 2.0,
          1.0, 2.0;

    _F = Eigen::MatrixXi(4, 3);
    _F << 0, 1, 2,
          1, 3, 2,
          2, 3, 5,
          2, 5, 4;
}

}

template <typename PassiveT>
void test_2d_deformation_performance(
        const std::string& _floating_type_str)
{
    Eigen::MatrixX<PassiveT> V_rest;
    Eigen::MatrixXi F_orig;
    planar_test_mesh(V_rest, F_orig);

    // Init stretched version
    Eigen::MatrixX<PassiveT> V = V_rest;
    V.col(0) *= 0.5;
    V.col(1) *= 0.25;

    // Duplicate faces
    Eigen::MatrixXi F = F_orig.replicate(1000 / F_orig.rows(), 1);

    // Optimize symmetric Dirichlet energy
    // 6 2D variables, 4 elements using 3 variables each.
    auto func = TinyAD::scalar_function<2, PassiveT>(TinyAD::range(V.rows()));

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

        return ((M * Mr.inverse()).squaredNorm() + (Mr * M.inverse()).squaredNorm()) / F.rows();
    });

    // Assemble initial x vector
    Eigen::VectorX<PassiveT> x = func.x_from_data([&] (int v_idx) { return V.row(v_idx); });

    // Eval derivatives
    {
        TinyAD::Timer timer(std::string(__FUNCTION__) + " evaluating " + std::to_string(F.rows()) + " elements (" + _floating_type_str + ")");
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
    }
}

TEST(PerformanceTest, 2DDeformationPerformanceFloat) { test_2d_deformation_performance<float>("float"); }
TEST(PerformanceTest, 2DDeformationPerformanceDouble) { test_2d_deformation_performance<double>("double"); }
TEST(PerformanceTest, 2DDeformationPerformanceLongDouble) { test_2d_deformation_performance<long double>("long double"); }
