/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include "Meshes.hh"
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/VectorFunction.hh>

template <typename PassiveT>
void test_compilation_time()
{
    // Load test mesh
    Eigen::MatrixX<PassiveT> V_rest;
    Eigen::MatrixX<PassiveT> V;
    Eigen::MatrixXi F;
    std::vector<Eigen::Index> b;
    std::vector<Eigen::Vector2<PassiveT>> bc;
    planar_test_mesh(V_rest, V, F, b, bc);

    // 2D variables
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

    // Don't call any of the eval_* functions.
    // After introducing deferred lambda instantiation, this should compile very fast.
}

TEST(DeferredLambdaTest, 2DDeformationFloat) { test_compilation_time<float>(); }
TEST(DeferredLambdaTest, 2DDeformationDouble) { test_compilation_time<double>(); }
TEST(DeferredLambdaTest, 2DDeformationLongDouble) { test_compilation_time<long double>(); }