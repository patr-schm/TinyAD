/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>

TEST(EigenTest, Auto)
{
    auto func = TinyAD::scalar_function<1>(TinyAD::range(1));
    func.add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // auto v = element.variables(0);
        // auto nv = -v.normalized();
        // return nv[0];

        // The above line fails because an expression template keeps a reference to temporary type.
        // Never use auto on the left-hand side of Eigen expressions.
        // See https://eigen.tuxfamily.org/dox/TopicPitfalls.html#TopicPitfalls_auto_keyword.
        //
        // Instead, write:

        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 1> v = element.variables(0);
        Eigen::Vector<T, 1> nv = -v.normalized();
        return nv[0];
    });

    const double f = func.eval(Eigen::Vector<double, 1>(1.0));
    const double eps = 1e-9;
    ASSERT_NEAR(f, -1.0, eps);
}
