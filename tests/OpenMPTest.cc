/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>

TEST(OpenMPTest, OpenMPEnabled)
{
    // Check if OpenMP is enabled
    #ifdef _OPENMP
        SUCCEED() << "OpenMP is enabled.";
    #else
        FAIL() << "OpenMP is not enabled.";
    #endif
}

TEST(OpenMPTest, ScalarFunctionParallel)
{
    auto func = TinyAD::scalar_function<1>(TinyAD::range(1));
    func.add_elements<1>(TinyAD::range(20), [] (auto& element)
    {
        // Assert parallel
        TINYAD_ASSERT_GEQ(omp_get_max_threads(), 2);

        return element.variables(0)[0];
    });

    func.eval_with_hessian_proj(Eigen::VectorXd::Zero(1));
}
