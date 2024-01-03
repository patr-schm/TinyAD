/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/GaussNewtonDirection.hh>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

namespace
{

template <typename PassiveT, typename SolverT>
void test_solver(
        const PassiveT& _eps)
{
    // x^2 + y^2 at x = (1, 1)
    // Test Newton step
    Eigen::VectorX<PassiveT> g(2);
    g.setConstant(2.0);
    Eigen::SparseMatrix<PassiveT> H = TinyAD::identity<PassiveT>(2);
    H *= 2.0;

    Eigen::SparseMatrix<PassiveT> B = TinyAD::identity<PassiveT>(2);

    SolverT solver;
    Eigen::VectorX<PassiveT> d = newton_direction(g, H, solver);
    ASSERT_NEAR(d(0), -1.0, _eps);
    ASSERT_NEAR(d(1), -1.0, _eps);

    d = newton_direction_reduced_basis(g, H, B, solver);
    ASSERT_NEAR(d(0), -1.0, _eps);
    ASSERT_NEAR(d(1), -1.0, _eps);

    // r = (x^2, y^2) at x = (1, 1)
    // Test Gauss-Newton step
    Eigen::VectorX<PassiveT> r(2);
    r.setConstant(1.0);
    Eigen::SparseMatrix<PassiveT> J = H;

    d = gauss_newton_direction(r, J, solver);
    ASSERT_NEAR(d(0), -0.5, _eps);
    ASSERT_NEAR(d(1), -0.5, _eps);
}

}

template <typename PassiveT>
void test_switch_solver(
            const PassiveT& _eps)
{
    test_solver<PassiveT, TinyAD::LinearSolver<PassiveT>>(_eps);
    test_solver<PassiveT, TinyAD::LinearSolver<PassiveT, Eigen::SimplicialLDLT<Eigen::SparseMatrix<PassiveT>>>>(_eps);
    test_solver<PassiveT, TinyAD::LinearSolver<PassiveT, Eigen::SimplicialLLT<Eigen::SparseMatrix<PassiveT>>>>(_eps);
    test_solver<PassiveT, TinyAD::LinearSolver<PassiveT, Eigen::SparseLU<Eigen::SparseMatrix<PassiveT>>>>(_eps);
    test_solver<PassiveT, TinyAD::LinearSolver<PassiveT, Eigen::SparseQR<Eigen::SparseMatrix<PassiveT>, Eigen::COLAMDOrdering<int>>>>(_eps);
}

TEST(SwitchSolverTest, 2DDeformationFloat) { test_switch_solver<float>(1e-6f); }
TEST(SwitchSolverTest, 2DDeformationDouble) { test_switch_solver<double>(1e-15); }
TEST(SwitchSolverTest, 2DDeformationLongDouble) { test_switch_solver<long double>(1e-15); }
