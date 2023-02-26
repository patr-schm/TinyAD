/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Operations/SVD.hh>

template <typename PassiveT, bool with_hessian>
void test_svd(
        const PassiveT _eps0,
        const PassiveT _eps1,
        const PassiveT _eps2)
{
    // Choose autodiff scalar type for 4 variables
    using ADouble = TinyAD::Scalar<4, PassiveT, with_hessian>;

    Eigen::Matrix2<PassiveT> A_passive = Eigen::Matrix2<PassiveT>::Random(2, 2);
    TINYAD_DEBUG_VAR(A_passive);

    Eigen::Matrix2<ADouble> A_active;
    A_active << ADouble(A_passive(0, 0), 0), ADouble(A_passive(0, 1), 1),
                ADouble(A_passive(1, 0), 2), ADouble(A_passive(1, 1), 3);

    // Eigen Jacobi SVD
    Eigen::JacobiSVD<Eigen::Matrix2<ADouble>> svd(A_active, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2<ADouble> U_eigen = svd.matrixU();
    Eigen::Vector2<ADouble> S_eigen = svd.singularValues();
    Eigen::Matrix2<ADouble> V_eigen = svd.matrixV();
    Eigen::Matrix2<ADouble> A_eigen = U_eigen * S_eigen.asDiagonal() * V_eigen.transpose();
    TINYAD_ASSERT_L((TinyAD::to_passive(A_eigen) - A_passive).array().abs().maxCoeff(), _eps0);

    // TinyAD closed-form SVD
    Eigen::Matrix2<ADouble> U_closed, V_closed;
    Eigen::Vector2<ADouble> S_closed;
    TinyAD::svd(A_active, U_closed, S_closed, V_closed);
    Eigen::Matrix2<ADouble> A_closed = U_closed * S_closed.asDiagonal() * V_closed.transpose();
    TINYAD_ASSERT_L((TinyAD::to_passive(A_closed) - A_passive).array().abs().maxCoeff(), _eps0);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            TINYAD_ASSERT_L(std::fabs(A_closed(i, j).val - A_eigen(i, j).val), _eps0);
            TINYAD_ASSERT_L((A_closed(i, j).grad - A_eigen(i, j).grad).array().abs().maxCoeff(), _eps1);
            if constexpr (with_hessian)
                TINYAD_ASSERT_L((A_closed(i, j).Hess - A_eigen(i, j).Hess).array().abs().maxCoeff(), _eps2);
        }
    }
}

TEST(SVDTest, SVDFloatFirstOrder) { test_svd<float, false>(1e-6f, 1e-6f, 1e-4f); }
TEST(SVDTest, SVDDoubleFirstOrder) { test_svd<double, false>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, SVDLongDoubleFirstOrder) { test_svd<long double, false>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, SVDFloatSecondOrder) { test_svd<float, true>(1e-6f, 1e-6f, 1e-3f); }
TEST(SVDTest, SVDDoubleSecondOrder) { test_svd<double, true>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, SVDLongDoubleSecondOrder) { test_svd<long double, true>(1e-12, 1e-8, 1e-4); }

template <typename PassiveT, bool with_hessian>
void test_closest_orthogonal(
        const PassiveT _eps0,
        const PassiveT _eps1,
        const PassiveT _eps2)
{
    // Choose autodiff scalar type for 4 variables
    using ADouble = TinyAD::Scalar<4, PassiveT, with_hessian>;

    Eigen::Matrix2<PassiveT> A_passive = Eigen::Matrix2<PassiveT>::Random(2, 2);
    TINYAD_DEBUG_VAR(A_passive);

    Eigen::Matrix2<ADouble> A_active;
    A_active << ADouble(A_passive(0, 0), 0), ADouble(A_passive(0, 1), 1),
                ADouble(A_passive(1, 0), 2), ADouble(A_passive(1, 1), 3);

    // Eigen Jacobi SVD
    Eigen::JacobiSVD<Eigen::Matrix2<ADouble>> svd(A_active, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2<ADouble> R_eigen = svd.matrixU() * svd.matrixV().transpose();

    // TinyAD closed-form SVD
    Eigen::Matrix2<ADouble> R_closed = closest_orthogonal(A_active);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            TINYAD_ASSERT_L(std::fabs(R_closed(i, j).val - R_eigen(i, j).val), _eps0);
            TINYAD_ASSERT_L((R_closed(i, j).grad - R_eigen(i, j).grad).array().abs().maxCoeff(), _eps1);
            if constexpr (with_hessian)
                TINYAD_ASSERT_L((R_closed(i, j).Hess - R_eigen(i, j).Hess).array().abs().maxCoeff(), _eps2);
        }
    }
}

TEST(SVDTest, ClosestOrthogonalFloatFirstOrder) { test_closest_orthogonal<float, false>(1e-6f, 1e-6f, 1e-4f); }
TEST(SVDTest, ClosestOrthogonalDoubleFirstOrder) { test_closest_orthogonal<double, false>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, ClosestOrthogonalLongDoubleFirstOrder) { test_closest_orthogonal<long double, false>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, ClosestOrthogonalFloatSecondOrder) { test_closest_orthogonal<float, true>(1e-6f, 1e-6f, 1e-3f); }
TEST(SVDTest, ClosestOrthogonalDoubleSecondOrder) { test_closest_orthogonal<double, true>(1e-12, 1e-8, 1e-4); }
TEST(SVDTest, ClosestOrthogonalLongDoubleSecondOrder) { test_closest_orthogonal<long double, true>(1e-12, 1e-8, 1e-4); }
