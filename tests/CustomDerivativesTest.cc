/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>

using ADouble = TinyAD::Double<1>;

double plus(const double a, const double b)
{
    return a + b;
}

ADouble plus(const ADouble& a, const ADouble& b)
{
    return ADouble::known_derivatives(
        a.val + b.val,
        a.grad + b.grad,
        a.Hess + b.Hess);
}

TEST(CustomDerivativesTest, Plus)
{
    // a(x) = x^2 + x + 2 at x=1
    // b(x) = x^3 - x^2 at x=1
    ADouble a = ADouble::known_derivatives(4.0, 3.0, 2.0);
    ADouble b = ADouble::known_derivatives(0.0, 1.0, 4.0);

    const ADouble f = plus(a, b);
    ASSERT_NEAR(f.val, 4.0, 1e-12);
    ASSERT_NEAR(f.grad(0), 4.0, 1e-12);
    ASSERT_NEAR(f.Hess(0, 0), 6.0, 1e-12);
}
