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

TEST(ElementTest, VariablesPassive)
{
    auto sf = TinyAD::scalar_function<1>(TinyAD::range(2));
    sf.add_elements<1>(TinyAD::range(1), [] (auto& element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T,1> a = element.variables_passive(1);
        Eigen::Vector<T,1> b = element.variables_passive(1);
        Eigen::Vector<T,1> c = element.variables_passive(1);
        return element.variable(0);
    });
    sf.eval(Eigen::Vector2d::Zero());
    sf.eval_with_derivatives(Eigen::Vector2d::Zero());

    auto vf = TinyAD::vector_function<1>(TinyAD::range(2));
    vf.add_elements<1, 1>(TinyAD::range(1), [] (auto& element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T,1> a = element.variables_passive(1);
        Eigen::Vector<T,1> b = element.variables_passive(1);
        Eigen::Vector<T,1> c = element.variables_passive(1);
        return Eigen::Vector<T,1> (element.variable(0));
    });
    vf.eval(Eigen::Vector2d::Zero());
    vf.eval_with_jacobian(Eigen::Vector2d::Zero());

}
