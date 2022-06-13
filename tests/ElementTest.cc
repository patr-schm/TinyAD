/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/VectorFunction.hh>

TEST(ElementTest, ElementTest)
{
    auto sf = TinyAD::scalar_function<1>(TinyAD::range(1));
    auto vf = TinyAD::vector_function<1>(TinyAD::range(1));

    // auto reference. Should compile
    sf.add_elements<1>(TinyAD::range(1), [] (auto& element) { return 0.0; });

    // auto value. Should NOT compile
//    sf.add_elements<1>(TinyAD::range(1), [] (auto element) { return 0.0; });

    // auto reference. Should compile
    vf.add_elements<1, 2>(TinyAD::range(1), [] (auto& element) { return Eigen::Vector2d(0.0, 0.0); });

    // auto value. Should NOT compile
//    vf.add_elements<1, 2>(TinyAD::range(1), [] (auto element) { return Eigen::Vector2d(0.0, 0.0); });
}
