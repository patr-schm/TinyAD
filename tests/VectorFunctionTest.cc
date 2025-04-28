/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/VectorFunction.hh>

// Simple test that verifies the vector function works correctly
TEST(VectorFunctionTest, Basic)
{
    // R^2 -> R^3
    // 2D input space, 2D+1D output (split across two element types)
    auto func = TinyAD::vector_function<1, double>(TinyAD::range(2));
    
    // First element: R^1 -> R^2
    func.template add_elements<1, 2>(TinyAD::range(1), [](auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x0 = element.variables(0)[0];
        return Eigen::Vector2<T>(2.0 * x0, sqr(x0));
    });
    
    // Second element: R^1 -> R^1 
    func.template add_elements<1, 1>(TinyAD::range(1), [](auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x1 = element.variables(1)[0];
        return Eigen::Vector<T, 1>(sqr(x1));
    });
    
    // Test at x = (3,4)
    Eigen::Vector2d x(3.0, 4.0);
    
    // Test function value
    Eigen::VectorXd r = func.eval(x);
    EXPECT_EQ(r.size(), 3);
    EXPECT_EQ(r[0], 6.0);    // 2*x0
    EXPECT_EQ(r[1], 9.0);    // x0^2
    EXPECT_EQ(r[2], 16.0);   // x1^2
    
    // Test sum of squares
    double f = func.eval_sum_of_squares(x);
    EXPECT_EQ(f, 373.0);     // 6^2 + 9^2 + 16^2
    
    // Test sum of squares with derivatives
    auto [f_s, g_s, r_s, J_s] = func.eval_sum_of_squares_with_derivatives(x);
    EXPECT_EQ(f_s, 373.0);
    EXPECT_EQ(r_s.size(), 3);
    EXPECT_EQ(r_s[0], 6.0);
    EXPECT_EQ(r_s[1], 9.0);
    EXPECT_EQ(r_s[2], 16.0);
    EXPECT_EQ(g_s[0], 8.0 * x[0] + 4.0 * x[0] * x[0] * x[0]);  // d/dx0(36 + 81 + 256)
    EXPECT_EQ(g_s[1], 4.0 * x[1] * x[1] * x[1]);               // d/dx1(36 + 81 + 256)
    
    // Test Jacobian
    auto [r_j, J] = func.eval_with_jacobian(x);
    EXPECT_EQ(r_j.size(), 3);
    EXPECT_EQ(J.rows(), 3);
    EXPECT_EQ(J.cols(), 2);
    EXPECT_EQ(J.coeff(0, 0), 2.0);     // d(2*x0)/dx0
    EXPECT_EQ(J.coeff(1, 0), 6.0);     // d(x0^2)/dx0
    EXPECT_EQ(J.coeff(2, 1), 8.0);     // d(x1^2)/dx1
    
    // Test Hessians
    auto [r_h, J_h, H] = func.eval_with_derivatives(x);
    EXPECT_EQ(r_h.size(), 3);
    EXPECT_EQ(H.size(), 3);
    EXPECT_EQ(H[1].coeff(0, 0), 2.0);  // d^2(x0^2)/dx0^2
    EXPECT_EQ(H[2].coeff(1, 1), 2.0);  // d^2(x1^2)/dx1^2
}

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
    Eigen::Vector2<PassiveT> g_expected((PassiveT)8.0 * x[0] + (PassiveT)4.0 * x[0] * x[0] * x[0], (PassiveT)4.0 * x[1] * x[1] * x[1]);
    Eigen::Vector3<PassiveT> r_expected((PassiveT)2.0 * x[0], sqr(x[0]), sqr(x[1]));
    Eigen::SparseMatrix<PassiveT> J_expected(3, 2);
    J_expected.coeffRef(0, 0) = (PassiveT)2.0;
    J_expected.coeffRef(1, 0) = (PassiveT)2.0 * x[0];
    J_expected.coeffRef(2, 1) = (PassiveT)2.0 * x[1];
    std::vector<Eigen::SparseMatrix<PassiveT>> H_expected(3, Eigen::SparseMatrix<PassiveT>(2, 2));
    H_expected[1].coeffRef(0, 0) = (PassiveT)2.0;
    H_expected[2].coeffRef(1, 1) = (PassiveT)2.0;

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

// Test that verifies move semantics work correctly for vector functions
TEST(VectorFunctionTest, Move)
{
    // R^2 -> R^3 (same as above)
    auto func1 = TinyAD::vector_function<1, double>(TinyAD::range(2));
    
    func1.template add_elements<1, 2>(TinyAD::range(1), [](auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x0 = element.variables(0)[0];
        return Eigen::Vector2<T>(2.0 * x0, sqr(x0));
    });
    
    func1.template add_elements<1, 1>(TinyAD::range(1), [](auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T x1 = element.variables(1)[0];
        return Eigen::Vector<T, 1>(sqr(x1));
    });
    
    // Create test point
    Eigen::Vector2d x(3.0, 4.0);
    
    // Evaluate function
    Eigen::VectorXd r1 = func1.eval(x);
    EXPECT_EQ(r1.size(), 3);
    EXPECT_EQ(r1[0], 6.0);    // 2*x0
    EXPECT_EQ(r1[1], 9.0);    // x0^2
    EXPECT_EQ(r1[2], 16.0);   // x1^2
    
    // Move the function
    auto func2 = std::move(func1);
    
    // Evaluate with Jacobian
    auto [r_j, J] = func2.eval_with_jacobian(x);
    EXPECT_EQ(r_j.size(), 3);
    EXPECT_EQ(J.coeff(0, 0), 2.0);     // d(2*x0)/dx0
    EXPECT_EQ(J.coeff(1, 0), 6.0);     // d(x0^2)/dx0
    EXPECT_EQ(J.coeff(2, 1), 8.0);     // d(x1^2)/dx1
    
    // Move again
    TinyAD::VectorFunction<1, double, Eigen::Index> func3;
    func3 = std::move(func2);
    
    // Evaluate with Hessians
    auto [r_h, J_h, H] = func3.eval_with_derivatives(x);
    EXPECT_EQ(r_h.size(), 3);
    EXPECT_EQ(J_h.coeff(0, 0), 2.0);     // d(2*x0)/dx0
    EXPECT_EQ(J_h.coeff(1, 0), 6.0);     // d(x0^2)/dx0
    EXPECT_EQ(J_h.coeff(2, 1), 8.0);     // d(x1^2)/dx1
    EXPECT_EQ(H[1].coeff(0, 0), 2.0);  // d^2(x0^2)/dx0^2
    EXPECT_EQ(H[2].coeff(1, 1), 2.0);  // d^2(x1^2)/dx1^2
}

// Test that verifies thread safety for vector functions
TEST(VectorFunctionTest, ThreadSafety)
{
    // Create a vector function
    auto func = TinyAD::vector_function<2, double>(TinyAD::range(1));
    
    // Add elements
    func.template add_elements<1, 2>(TinyAD::range(1), [](auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        auto x = element.variables(0);
        return Eigen::Vector2<T>(x[0] * x[0], x[1] * x[1]);
    });
    
    // Create test point
    Eigen::Vector2d x(1.0, 2.0);
    
    // Simulate concurrent access by calling eval methods multiple times
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            func.eval(x);
        }
        
        #pragma omp section
        {
            func.eval_with_jacobian(x);
        }
        
        #pragma omp section
        {
            func.eval_with_derivatives(x);
        }
    }
    
    // If we got here without crashes, the test passes
    SUCCEED();
}
