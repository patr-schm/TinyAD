/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <gtest/gtest.h>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/Helpers.hh>

template <typename T>
T f(
    const T& x1, const T& x2,
    const T& y1, const T& y2, const T& y3)
{
    // From https://github.com/patr-schm/TinyAD/issues/13
    const auto r1 = x1-y1;
    const auto r2 = x2-y2;
    const auto d = r1*r1 + r2*r2;
    return y3*d;
}

template <typename PassiveT>
void test_hess_block_1()
{
    PassiveT x1 = 1;
    PassiveT x2 = 2;
    PassiveT y1 = 3;
    PassiveT y2 = 4;
    PassiveT y3 = 5;

    // Compute full Hessian
    using ADFull = TinyAD::Scalar<5, PassiveT>;
    Eigen::Matrix<ADFull,5,1> x_full = ADFull::make_active({x1, x2, y1, y2, y3});
    ADFull f_ad_full = f(x_full[0], x_full[1], x_full[2], x_full[3], x_full[4]);

    // Compute truncated Hessian (2-by-3 block ddf/dxdy)
    using ADTrunc = TinyAD::Scalar<5, PassiveT, true, 0, 2, 2, 3>;
    Eigen::Matrix<ADTrunc, 5, 1> x_trunc = ADTrunc::make_active({x1, x2, y1, y2, y3});
    ADTrunc f_ad_trunc = f(x_trunc[0], x_trunc[1], x_trunc[2], x_trunc[3], x_trunc[4]);

    TINYAD_ASSERT_EPS_MAT(f_ad_trunc.Hess, f_ad_full.Hess.block(0, 2, 2, 3), 1e-16);
}

TEST(ScalarTestHessianBlock, HessianBlock1Float) { test_hess_block_1<float>(); }
TEST(ScalarTestHessianBlock, HessianBlock1Double) { test_hess_block_1<double>(); }
TEST(ScalarTestHessianBlock, HessianBlock1LongDouble) { test_hess_block_1<long double>(); }

template <typename T, typename PassiveT>
T symm_dirich(const Eigen::Matrix<T, 6, 1>& _x, const Eigen::Matrix<PassiveT, 6, 1>& _x_rest)
{
    const Eigen::Matrix<PassiveT, 2, 1> ar(_x_rest[0], _x_rest[1]);
    const Eigen::Matrix<PassiveT, 2, 1> br(_x_rest[2], _x_rest[3]);
    const Eigen::Matrix<PassiveT, 2, 1> cr(_x_rest[4], _x_rest[5]);
    const Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);

    const Eigen::Matrix<T, 2, 1> a(_x[0], _x[1]);
    const Eigen::Matrix<T, 2, 1> b(_x[2], _x[3]);
    const Eigen::Matrix<T, 2, 1> c(_x[4], _x[5]);
    const Eigen::Matrix<T, 2, 2> M = TinyAD::col_mat(b - a, c - a);

    const Eigen::Matrix<T, 2, 2> J = M * Mr.inverse();
    const PassiveT area = 1.0;

    return area * (J.squaredNorm() + J.inverse().squaredNorm());
}

template <typename PassiveT, int hess_row_start, int hess_col_start, int hess_rows, int hess_cols>
void test_hessian_block_symmetric_dirichlet()
{
    // Passive rest-state triangle ar, br, cr
    const Eigen::Matrix<PassiveT, 2, 1> ar(1.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> br(2.0, 1.0);
    const Eigen::Matrix<PassiveT, 2, 1> cr(1.0, 2.0);
    const Eigen::Matrix<PassiveT, 2, 2> Mr = TinyAD::col_mat(br - ar, cr - ar);

    const Eigen::Matrix<PassiveT, 6, 1> x_rest = { 1.0, 1.0, 2.0, 1.0, 1.0, 2.0 };
    const Eigen::Matrix<PassiveT, 6, 1> x_init = { 10.0, 1.0, 15.0, 3.0, 2.0, 2.0 };

    // Compute full 6-by-6 Hessian
    using ADFull = TinyAD::Scalar<6, PassiveT>;
    const Eigen::Matrix<PassiveT, 6, 6> H_full = symm_dirich(ADFull::make_active(x_init), x_rest).Hess;

    // Compute a Hessian block only and compare
    using ADBlock = TinyAD::Scalar<6, PassiveT, true, hess_row_start, hess_col_start, hess_rows, hess_cols>;
    const Eigen::Matrix<PassiveT, hess_rows, hess_cols> H_block = symm_dirich(ADBlock::make_active(x_init), x_rest).Hess;

    TINYAD_ASSERT_EPS_MAT(H_block, H_full.block(hess_row_start, hess_col_start, hess_rows, hess_cols), 1e-16);
}

TEST(ScalarTestHessianBlock, HessianBlockSymmetricDirichletDouble)
{
    test_hessian_block_symmetric_dirichlet<double, 0, 0, 0, 0>();
    test_hessian_block_symmetric_dirichlet<double, 0, 0, 6, 6>();
    test_hessian_block_symmetric_dirichlet<double, 0, 0, 3, 1>();
    test_hessian_block_symmetric_dirichlet<double, 0, 0, 1, 3>();
    test_hessian_block_symmetric_dirichlet<double, 2, 2, 2, 2>();
    test_hessian_block_symmetric_dirichlet<double, 1, 4, 5, 2>();
}
