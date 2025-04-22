/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <complex>

template <typename PassiveT, bool with_hessian>
void test_complex(const PassiveT _eps)
{
    using ScalarT = TinyAD::Scalar<4, PassiveT, with_hessian>;

    std::complex<ScalarT> a(ScalarT(1, 0), ScalarT(2, 1));
    std::complex<ScalarT> b(ScalarT(3, 2), ScalarT(4, 3));

    {   // Addition
        std::complex<ScalarT> c = a + b;

        TINYAD_ASSERT_EPS(c.real().val, 4.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().val, 6.0, _eps);
    }

    {   // Subtraction
        std::complex<ScalarT> c = a - b;

        TINYAD_ASSERT_EPS(c.real().val, -2.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().val, -2.0, _eps);
    }

    {   // Multiplication
        std::complex<ScalarT> c = a * b;

        TINYAD_ASSERT_EPS(c.real().val, -5.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().val, 10.0, _eps);

        TINYAD_ASSERT_EPS(c.real().grad(0), 3.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(0), 4.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(1), -4.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(1), 3.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(2), 1.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(2), 2.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(3), -2.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(3), 1.0, _eps);

        if constexpr (with_hessian)
        {
            Eigen::Matrix4<PassiveT> H_real;
            H_real << 0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, -1.0,
                      1.0, 0.0, 0.0, 0.0,
                      0.0, -1.0, 0.0, 0.0;
            Eigen::Matrix4<PassiveT> H_imag;
            H_imag << 0.0, 0.0, 0.0, 1.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      1.0, 0.0, 0.0, 0.0;

            TINYAD_ASSERT_L((c.real().Hess - H_real).cwiseAbs().maxCoeff(), _eps);
        }
    }

    {   // Division
        std::complex<ScalarT> c = a / b;

        TINYAD_ASSERT_EPS(c.real().val, 11.0 / 25.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().val, 2.0 / 25.0, _eps);

        TINYAD_ASSERT_EPS(c.real().grad(0), 3.0 / 25.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(0), -4.0 / 25.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(1), 4.0 / 25.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(1), 3.0 / 25.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(2), -41.0 / 625.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(2), 38.0 / 625.0, _eps);
        TINYAD_ASSERT_EPS(c.real().grad(3), -38.0 / 625.0, _eps);
        TINYAD_ASSERT_EPS(c.imag().grad(3), -41.0 / 625.0, _eps);

        // TODO: Check Hessian
    }

    {   // Conjugate
        std::complex<ScalarT> c = conj(a);

        TINYAD_ASSERT_EQ(c.real().val, a.real().val);
        TINYAD_ASSERT_EQ(c.imag().val, -a.imag().val);

        TINYAD_ASSERT_EQ(c.real().grad, a.real().grad);
        TINYAD_ASSERT_EQ(c.imag().grad, -a.imag().grad);

        if constexpr (with_hessian)
        {
            TINYAD_ASSERT_EQ(c.real().Hess, a.real().Hess);
            TINYAD_ASSERT_EQ(c.imag().Hess, -a.imag().Hess);
        }
    }

    {   // Abs
        ScalarT length = abs(a);
        ScalarT length_ref = hypot(a.real(), a.imag());

        TINYAD_ASSERT_EPS(length.val, length_ref.val, _eps);
        TINYAD_ASSERT_L((length.grad - length_ref.grad).cwiseAbs().maxCoeff(), _eps);

        if constexpr (with_hessian)
            TINYAD_ASSERT_L((length.Hess - length_ref.Hess).cwiseAbs().maxCoeff(), _eps);
    }

    {   // Arg
        ScalarT angle = arg(a);
        ScalarT angle_ref = atan2(a.imag(), a.real());

        TINYAD_ASSERT_EPS(angle.val, angle_ref.val, _eps);
        TINYAD_ASSERT_L((angle.grad - angle_ref.grad).cwiseAbs().maxCoeff(), _eps);

        if constexpr (with_hessian)
            TINYAD_ASSERT_L((angle.Hess - angle_ref.Hess).cwiseAbs().maxCoeff(), _eps);
    }
}

TEST(ComplexTest, ComplexFloatFirstOrder) { test_complex<float, false>(1e-6f); }
TEST(ComplexTest, ComplexDoubleFirstOrder) { test_complex<double, false>(1e-12); }
TEST(ComplexTest, ComplexLongDoubleFirstOrder) { test_complex<long double, false>(1e-15); }
TEST(ComplexTest, ComplexFloatSecondOrder) { test_complex<float, true>(1e-6f); }
TEST(ComplexTest, ComplexDoubleSecondOrder) { test_complex<double, true>(1e-12); }
TEST(ComplexTest, ComplexLongDoubleSecondOrder) { test_complex<long double, true>(1e-15); }
