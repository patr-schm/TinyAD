/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

/**
 * Compute update vector d such that x + d performs a Newton step
 * (i.e. minimizes the quadratic approximation at x).
 * Input:
 *      _g: gradient
 *      _H_proj: symmetric positive-definite Hessian approximation
 *      _solver: A solver that can be used over multiple iterations
 *               in case the sparsity pattern of _H_proj is constant.
 */
template <typename PassiveT, typename SolverT>
Eigen::VectorX<PassiveT> newton_direction(
        const Eigen::VectorX<PassiveT>& _g,
        const Eigen::SparseMatrix<PassiveT>& _H_proj,
        LinearSolver<PassiveT, SolverT>& _solver,
        const PassiveT& _w_identity = 0.0)
{
    const Eigen::SparseMatrix<PassiveT> H_reg = _w_identity * identity<PassiveT>(_g.size()) + _H_proj;

    if (_solver.sparsity_pattern_dirty)
    {
        _solver.solver.analyzePattern(H_reg);
        _solver.sparsity_pattern_dirty = false;
    }

    _solver.solver.factorize(H_reg);
    const Eigen::VectorX<PassiveT> d = _solver.solver.solve(-_g);

    if (_solver.solver.info() != Eigen::Success)
        TINYAD_ERROR_throw("Linear solve failed.");

    TINYAD_ASSERT_FINITE_MAT(d);
    return d;
}

/**
 * Compute update vector d such that x + d performs a Newton step
 * (i.e. minimizes the quadratic approximation at x).
 * Input:
 *      _g: gradient
 *      _H_proj: symmetric positive definite Hessian approximation
 */
template <typename PassiveT>
Eigen::VectorX<PassiveT> newton_direction(
        const Eigen::VectorX<PassiveT>& _g,
        const Eigen::SparseMatrix<PassiveT>& _H_proj,
        const PassiveT& _w_identity = 0.0)
{
    LinearSolver<PassiveT> solver;
    return newton_direction(_g, _H_proj, solver, _w_identity);
}

/**
 * Compute update vector d such that x + d performs a Newton step
 * (i.e. minimizes the quadratic approximation at x),
 * constrained to a linear subspace with known basis.
 * The n-by-m matrix B maps from the subspace to the solution space,
 * (i.e. d = B * d_reduced).
 * For problems with constant sparsity pattern, cache _solver
 * to benefit from pre-factorization.
 */
template <typename PassiveT, typename SolverT>
Eigen::VectorX<PassiveT> newton_direction_reduced_basis(
        const Eigen::VectorX<PassiveT>& _g,
        const Eigen::SparseMatrix<PassiveT>& _H_proj,
        const Eigen::SparseMatrix<PassiveT>& _B,
        LinearSolver<PassiveT, SolverT>& _solver,
        const PassiveT& _w_identity = 0.0)
{
    const Eigen::Index n = _B.rows();
    const Eigen::Index m = _B.cols();
    TINYAD_ASSERT_EQ(_g.rows(), n);
    TINYAD_ASSERT_EQ(_H_proj.rows(), n);
    TINYAD_ASSERT_EQ(_H_proj.cols(), n);

    const Eigen::SparseMatrix<PassiveT> H_reduced = _B.transpose() * _H_proj * _B + _w_identity * identity<PassiveT>(m);
    if (_solver.sparsity_pattern_dirty)
    {
        _solver.solver.analyzePattern(H_reduced);
        _solver.sparsity_pattern_dirty = false;
    }

    _solver.solver.factorize(H_reduced);
    const Eigen::VectorX<PassiveT> d_reduced = _solver.solver.solve(-_B.transpose() * _g);

    if (_solver.solver.info() != Eigen::Success)
        TINYAD_ERROR_throw("Linear solve failed.");

    TINYAD_ASSERT_FINITE_MAT(d_reduced);
    return _B * d_reduced;
}

/**
 * Compute update vector d such that x + d performs a Newton step
 * (i.e. minimizes the quadratic approximation at x),
 * constrained to a linear subspace with known basis.
 * The n-by-m matrix B maps from the subspace to the solution space,
 * (i.e. d = B * d_reduced).
 */
template <typename PassiveT>
Eigen::VectorX<PassiveT> newton_direction_reduced_basis(
        const Eigen::VectorX<PassiveT>& _g,
        const Eigen::SparseMatrix<PassiveT>& _H_proj,
        const Eigen::SparseMatrix<PassiveT>& _B,
        const PassiveT& _w_identity = 0.0)
{
    LinearSolver<PassiveT> solver;
    return newton_direction_reduced_basis(_g, _H_proj, _B, solver, _w_identity);
}

}
