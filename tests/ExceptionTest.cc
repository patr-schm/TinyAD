/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>

TEST(ExceptionTest, ExceptionTest)
{
    const int n_elements = 10;
    bool throw_exception = false;

    using Vector1 = Eigen::Matrix<double, 1, 1>;
    auto func = TinyAD::scalar_function<1>(TinyAD::range(1));
    func.template add_elements<1>(TinyAD::range(n_elements), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        element.variables(0);
        
        if (throw_exception)
            throw std::runtime_error("Exception in parallel section");
        else
            return 1.0;
    });

    throw_exception = true;
    bool caught_first_exception = false;
    bool caught_second_exception = false;

    try
    {
        auto [f, g, H] = func.eval_with_hessian_proj(Vector1(0.0));
    }
    catch (const std::exception&)
    {
        caught_first_exception = true;
    }

    try
    {
        auto [f, g, H] = func.eval_with_hessian_proj(Vector1(0.0));
    }
    catch (const std::exception&)
    {
        caught_second_exception = true;
    }

    TINYAD_ASSERT(caught_first_exception);
    TINYAD_ASSERT(caught_second_exception);

    // Make sure we can still proceed normally
    throw_exception = false;
    auto [f, g, H] = func.eval_with_hessian_proj(Vector1(0.0));
    TINYAD_ASSERT_EPS(f, n_elements, 1e-12);
}
