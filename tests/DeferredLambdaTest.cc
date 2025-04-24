/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt, claude-3.7-sonnet
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>

// Simple test that verifies the function works correctly
TEST(DeferredLambdaTest, BasicFunctionality) {
    // Create a scalar function
    auto func = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func.template add_elements<1>(TinyAD::range(1), [](auto& element) {
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

// Test that verifies move semantics work correctly
TEST(DeferredLambdaTest, MoveSemantics) {
    // Create a scalar function
    auto func1 = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func1.template add_elements<1>(TinyAD::range(1), [](auto& element) {
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

// Test that verifies thread safety
TEST(DeferredLambdaTest, ThreadSafety) {
    // Create a scalar function
    auto func = TinyAD::scalar_function<2>(TinyAD::range(1));
    
    // Add a simple quadratic function
    func.template add_elements<1>(TinyAD::range(1), [](auto& element) {
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