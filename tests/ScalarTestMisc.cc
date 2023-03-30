/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_quadratic()
{
    // f(a) = a^2 + a + 2 at a=1
    constexpr int dim = dynamic ? Eigen::Dynamic : 1;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    const ADouble a = ADouble::make_active(1.0, 0, 1);
    const ADouble f = sqr(a) + a + 2.0;
    ASSERT_EQ(f.val, 4.0);
    ASSERT_EQ(f.grad(0), 3.0);
    if constexpr (with_hessian)
    {
        ASSERT_EQ(f.Hess(0, 0), 2.0);
        TINYAD_ASSERT_SYMMETRIC(f.Hess, 1e-12);
    }
}

TEST(ScalarTestMisc, QuadraticFloatFirstOrder) { test_quadratic<float, false>();}
TEST(ScalarTestMisc, QuadraticDoubleFirstOrder) { test_quadratic<double, false>();}
TEST(ScalarTestMisc, QuadraticLongDoubleFirstOrder) { test_quadratic<long double, false>();}
TEST(ScalarTestMisc, QuadraticFloatSecondOrder) { test_quadratic<float, true>();}
TEST(ScalarTestMisc, QuadraticDoubleSecondOrder) { test_quadratic<double, true>();}
TEST(ScalarTestMisc, QuadraticLongDoubleSecondOrder) { test_quadratic<long double, true>();}

TEST(ScalarTestMisc, QuadraticDoubleFirstOrderDynamic) { test_quadratic<double, false, true>(); }
TEST(ScalarTestMisc, QuadraticDoubleSecondOrderDynamic) { test_quadratic<double, true, true>(); }

template <typename PassiveT, bool with_hessian, bool dynamic = false>
void test_sphere()
{
    // f: R^2 -> R^3
    // f(phi, psi) = (sin(phi) * cos(psi), sin(phi) * sin(psi), cos(phi))
    constexpr int dim = dynamic ? Eigen::Dynamic : 2;
    using ADouble = TinyAD::Scalar<dim, PassiveT, with_hessian>;
    ADouble alpha = ADouble::make_active((PassiveT)M_PI / 8.0, 0, 2);
    ADouble beta = ADouble::make_active((PassiveT)M_PI / 8.0, 1, 2);
    const auto f = Eigen::Matrix<ADouble, 3, 1>(
            sin(alpha) * cos(beta),
            sin(alpha) * sin(beta),
            cos(alpha));

    // Test function value
    ASSERT_NEAR(f[0].val, std::sin(alpha.val) * std::cos(alpha.val), 1e-12);
    ASSERT_NEAR(f[1].val, std::sin(alpha.val) * std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].val, std::cos(alpha.val), 1e-12);

    // Test gradient (Jacobian)
    ASSERT_NEAR(f[0].grad(0), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
    ASSERT_NEAR(f[0].grad(1), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
    ASSERT_NEAR(f[1].grad(0), std::cos(alpha.val) * std::sin(beta.val), 1e-12);
    ASSERT_NEAR(f[1].grad(1), std::cos(beta.val) * std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].grad(0), -std::sin(alpha.val), 1e-12);
    ASSERT_NEAR(f[2].grad(1), 0.0, 1e-12);

    if constexpr (with_hessian)
    {
        // Test Hessian
        ASSERT_NEAR(f[0].Hess(0, 0), -std::sin(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(0, 1), -std::cos(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(1, 0), -std::cos(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[0].Hess(1, 1), -std::sin(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(0, 0), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(0, 1), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(1, 0), std::cos(alpha.val) * std::cos(beta.val), 1e-12);
        ASSERT_NEAR(f[1].Hess(1, 1), -std::sin(alpha.val) * std::sin(beta.val), 1e-12);
        ASSERT_NEAR(f[2].Hess(0, 0), -std::cos(alpha.val), 1e-12);
        ASSERT_NEAR(f[2].Hess(0, 1), 0.0, 1e-12);
        ASSERT_NEAR(f[2].Hess(1, 0), 0.0, 1e-12);
        ASSERT_NEAR(f[2].Hess(1, 1), 0.0, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[0].Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[1].Hess, 1e-12);
        TINYAD_ASSERT_SYMMETRIC(f[2].Hess, 1e-12);
    }
}

TEST(ScalarTestMisc, SphereFloatFirstOrder) { test_sphere<float, false>(); }
TEST(ScalarTestMisc, SphereDoubleFirstOrder) { test_sphere<double, false>(); }
TEST(ScalarTestMisc, SphereLongDoubleFirstOrder) { test_sphere<long double, false>(); }
TEST(ScalarTestMisc, SphereFloatSecondOrder) { test_sphere<float, true>(); }
TEST(ScalarTestMisc, SphereDoubleSecondOrder) { test_sphere<double, true>(); }
TEST(ScalarTestMisc, SphereLongDoubleSecondOrder) { test_sphere<long double, true>(); }

TEST(ScalarTestMisc, SphereDoubleFirstOrderDynamic) { test_sphere<double, false, true>(); }
TEST(ScalarTestMisc, SphereDoubleSecondOrderDynamic) { test_sphere<double, true, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_min_quadric()
{
    // Variable vector in R^3
    constexpr int dim = dynamic ? Eigen::Dynamic : 3;
    using ADouble = TinyAD::Scalar<dim, PassiveT>;
    const Eigen::Vector<ADouble, dim> x = ADouble::make_active({ 0.0, 0.0, 0.0 });

    // Quadratic function
    const ADouble f = sqr(x[0]) + 2.0 * sqr(x[1]) + 6.0 * sqr(x[2]) + x[0] - 2.0 * x[1] + 6.0 * x[2] + 10;

    // Solve for minimum
    const Eigen::Vector<PassiveT, dim> x_min = -f.Hess.inverse() * f.grad;

    ASSERT_NEAR(x_min.x(), -0.5, 1e-12);
    ASSERT_NEAR(x_min.y(), 0.5, 1e-12);
    ASSERT_NEAR(x_min.z(), -0.5, 1e-12);
}

TEST(ScalarTestMisc, MinQuadraticFloat) { test_min_quadric<float>(); }
TEST(ScalarTestMisc, MinQuadraticDouble) { test_min_quadric<double>(); }
TEST(ScalarTestMisc, MinQuadraticLongDouble) { test_min_quadric<long double>(); }

TEST(ScalarTestMisc, MinQuadraticDoubleDynamic) { test_min_quadric<double, true>(); }

template <typename PassiveT, bool dynamic = false>
void test_triangle_distortion()
{
    constexpr int dim = dynamic ? Eigen::Dynamic : 6;
    using ADouble = TinyAD::Scalar<dim, PassiveT>;

    // Passive rest-state triangle ar, br, cr
    const Eigen::Matrix<PassiveT, 2, 1> ar(1.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> br(2.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> cr(1.0, 2.0);
    const Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);

    // Active variable vector for vertex positions a, b, c
    const Eigen::Vector<ADouble, dim> x = ADouble::make_active({
        10.0, 1.0,
        15.0, 3.0,
        2.0, 2.0,
    });
    const Eigen::Matrix<ADouble, 2, 1> a(x[0], x[1]);
    const Eigen::Matrix<ADouble, 2, 1> b(x[2], x[3]);
    const Eigen::Matrix<ADouble, 2, 1> c(x[4], x[5]);
    const Eigen::Matrix<ADouble, 2, 2> M = TinyAD::col_mat(b - a, c - a);

    const Eigen::Matrix<ADouble, 2, 2> J = M * Mr.inverse();
    const ADouble E = J.squaredNorm() + J.inverse().squaredNorm();
    TINYAD_ASSERT_FINITE(E.val);
    TINYAD_ASSERT_FINITE_MAT(E.grad);
    TINYAD_ASSERT_FINITE_MAT(E.Hess);
}

TEST(ScalarTestMisc, TriangleDistortionFloat) { test_triangle_distortion<float>(); }
TEST(ScalarTestMisc, TriangleDistortionDouble) { test_triangle_distortion<double>(); }
TEST(ScalarTestMisc, TriangleDistortionLongDouble) { test_triangle_distortion<long double>(); }

//TEST(ScalarTestMisc, TriangleDistortionDoubleDynamic) { test_triangle_distortion<double, true>(); } // Not available, b/c squaredNorm() needs default constructor "Scalar(0)" to be implemented, which is currently not the case in dynamic mode.
