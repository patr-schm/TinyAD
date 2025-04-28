/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt, claude-3.7-sonnet
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/VectorFunction.hh>

// Simple test that verifies the scalar function works correctly
TEST(DeferredLambdaTest, BasicFunctionalityScalar)
{
    // Create a scalar function
    auto func = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func.template add_elements<1>(TinyAD::range(1), [](auto& element)
    {
        auto x = element.variables(0);
        return x[0] * x[0] + x[1] * x[1];
    });
    
    // Create test point
    Eigen::Vector2d x(1.0, 2.0);
    
    // Evaluate function
    double f = func.eval(x);
    EXPECT_EQ(f, 5.0);
    
    // Evaluate function with gradient
    auto [f_g, g] = func.eval_with_gradient(x);
    EXPECT_EQ(f_g, 5.0);
    EXPECT_EQ(g[0], 2.0);
    EXPECT_EQ(g[1], 4.0);
    
    // Evaluate function with Hessian
    auto [f_h, g_h, H] = func.eval_with_derivatives(x);
    EXPECT_EQ(f_h, 5.0);
    EXPECT_EQ(g_h[0], 2.0);
    EXPECT_EQ(g_h[1], 4.0);
    EXPECT_EQ(H.coeff(0, 0), 2.0);
    EXPECT_EQ(H.coeff(1, 1), 2.0);
}

// Test that verifies move semantics work correctly for scalar functions
TEST(DeferredLambdaTest, MoveSemanticsScalar)
{
    // Create a scalar function
    auto func1 = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func1.template add_elements<1>(TinyAD::range(1), [](auto& element)
    {
        auto x = element.variables(0);
        return x[0] * x[0] + x[1] * x[1];
    });
    
    // Create test point
    Eigen::Vector2d x(1.0, 2.0);
    
    // Evaluate function
    double f1 = func1.eval(x);
    EXPECT_EQ(f1, 5.0);
    
    // Move the function
    auto func2 = std::move(func1);
    
    // Evaluate with gradient
    auto [f_g, g] = func2.eval_with_gradient(x);
    EXPECT_EQ(f_g, 5.0);
    EXPECT_EQ(g[0], 2.0);
    EXPECT_EQ(g[1], 4.0);
    
    // Move again
    TinyAD::ScalarFunction<2, double, Eigen::Index> func3;
    func3 = std::move(func2);
    
    // Evaluate with Hessian
    auto [f_h, g_h, H] = func3.eval_with_derivatives(x);
    EXPECT_EQ(f_h, 5.0);
    EXPECT_EQ(g_h[0], 2.0);
    EXPECT_EQ(g_h[1], 4.0);
    EXPECT_EQ(H.coeff(0, 0), 2.0);
    EXPECT_EQ(H.coeff(1, 1), 2.0);
}

// Test that verifies thread safety for scalar functions
TEST(DeferredLambdaTest, ThreadSafetyScalar)
{
    // Create a scalar function
    auto func = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func.template add_elements<1>(TinyAD::range(1), [](auto& element)
    {
        auto x = element.variables(0);
        return x[0] * x[0] + x[1] * x[1];
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
            func.eval_with_gradient(x);
        }
        
        #pragma omp section
        {
            func.eval_with_derivatives(x);
        }
    }
    
    // If we got here without crashes, the test passes
    SUCCEED();
}

// Simple test that verifies the vector function works correctly
TEST(DeferredLambdaTest, BasicFunctionalityVector)
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

// Test that verifies move semantics work correctly for vector functions
TEST(DeferredLambdaTest, MoveSemanticsVector)
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
TEST(DeferredLambdaTest, ThreadSafetyVector)
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