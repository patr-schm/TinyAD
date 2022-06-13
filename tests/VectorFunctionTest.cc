/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/VectorFunction.hh>

template <typename PassiveT>
void test_eval()
{
    // R^2 -> R^3
    // 2 1D variables, 1 2D element, 1 1D element
    auto func = TinyAD::vector_function<1, PassiveT>(TinyAD::range(2));
    func.template add_elements<1, 2>(TinyAD::range(1), [] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x0 = element.variables(0)[0];

        return Eigen::Vector2<T>(2.0 * x0, sqr(x0));
    });
    func.template add_elements<1, 1>(TinyAD::range(1), [] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x1 = element.variables(1)[0];

        return Eigen::Vector<T, 1>(sqr(x1));
    });

    Eigen::Vector2<PassiveT> x(3.0, 4.0);

    // (2.0 * x0, x0^2, x1^2)
    PassiveT f_expected = 373.0;
    Eigen::Vector2<PassiveT> g_expected(8.0 * x[0] + 4.0 * x[0] * x[0] * x[0], 4.0 * x[1] * x[1] * x[1]);
    Eigen::Vector3<PassiveT> r_expected(2.0 * x[0], sqr(x[0]), sqr(x[1]));
    Eigen::SparseMatrix<PassiveT> J_expected(3, 2);
    J_expected.coeffRef(0, 0) = 2.0;
    J_expected.coeffRef(1, 0) = 2.0 * x[0];
    J_expected.coeffRef(2, 1) = 2.0 * x[1];
    std::vector<Eigen::SparseMatrix<PassiveT>> H_expected(3, Eigen::SparseMatrix<PassiveT>(2, 2));
    H_expected[1].coeffRef(0, 0) = 2.0;
    H_expected[2].coeffRef(1, 1) = 2.0;

    {   // Test eval()
        Eigen::VectorX<PassiveT> r = func.eval(x);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
    }

    {   // Test eval_sum_of_squares()
        PassiveT f = func.eval_sum_of_squares(x);
        TINYAD_ASSERT_EPS(f, f_expected, 1e-16);
    }

    {   // Test eval_with_jacobian()
        Eigen::VectorX<PassiveT> r;
        Eigen::SparseMatrix<PassiveT> J;
        func.eval_with_jacobian(x, r, J);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
    }

    {   // Test eval_with_jacobian()
        auto [r, J] = func.eval_with_jacobian(x);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
    }

    {   // Test eval_with_derivatives()
        Eigen::VectorX<PassiveT> r;
        Eigen::SparseMatrix<PassiveT> J;
        std::vector<Eigen::SparseMatrix<PassiveT>> H;
        func.eval_with_derivatives(x, r, J, H);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
        TINYAD_ASSERT_EQ((int)H.size(), r.size());
        for (int i = 0; i < r.size(); ++i)
            TINYAD_ASSERT_EPS_MAT(H[i].toDense(), H_expected[i].toDense(), 1e-16);
    }

    {   // Test eval_with_derivatives()
        auto [r, J, H] = func.eval_with_derivatives(x);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
        TINYAD_ASSERT_EQ((int)H.size(), r.size());
        for (int i = 0; i < r.size(); ++i)
            TINYAD_ASSERT_EPS_MAT(H[i].toDense(), H_expected[i].toDense(), 1e-16);
    }

    {   // Test eval_sum_of_squares_with_derivatives()
        PassiveT f;
        Eigen::VectorX<PassiveT> g;
        Eigen::VectorX<PassiveT> r;
        Eigen::SparseMatrix<PassiveT> J;
        func.eval_sum_of_squares_with_derivatives(x, f, g, r, J);
        TINYAD_ASSERT_EPS(f, f_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(g, g_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
    }

    {   // Test eval_sum_of_squares_with_derivatives()
        auto [f, g, r, J] = func.eval_sum_of_squares_with_derivatives(x);
        TINYAD_ASSERT_EPS(f, f_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(g, g_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(r, r_expected, 1e-16);
        TINYAD_ASSERT_EPS_MAT(J.toDense(), J_expected.toDense(), 1e-16);
    }
}

TEST(VectorFunctionTest, 1DFloat) { test_eval<float>(); }
TEST(VectorFunctionTest, 1DDouble) { test_eval<double>(); }
TEST(VectorFunctionTest, 1DLongDouble) { test_eval<long double>(); }

