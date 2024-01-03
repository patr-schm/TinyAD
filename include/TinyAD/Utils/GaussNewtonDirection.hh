/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

/**
 * Compute update vector d such that x + d performs a Gauss-Newton step
 * on a sum-of-squares function f(x) = sum_i r_i(x)^2.
 * Input:
 *      _r: vector of residuals
 *      _J: Jacobian of residuals. Size n_outputs-by-n_vars
 *      _solver: A solver that can be used over multiple iterations
 *               in case the sparsity pattern of J^T * J is constant.
 */
template <typename PassiveT, typename SolverT>
Eigen::VectorX<PassiveT> gauss_newton_direction(
        const Eigen::VectorX<PassiveT>& _r,
        const Eigen::SparseMatrix<PassiveT>& _J,
        LinearSolver<PassiveT, SolverT>& _solver,
        const PassiveT& _w_identity = 0.0)
{
    const Eigen::SparseMatrix<PassiveT> JtJ_reg = _w_identity * identity<PassiveT>(_J.cols()) + _J.transpose() * _J;

    if (_solver.sparsity_pattern_dirty)
    {
        _solver.solver.analyzePattern(JtJ_reg);
        _solver.sparsity_pattern_dirty = false;
    }

    _solver.solver.factorize(JtJ_reg);
    const Eigen::VectorX<PassiveT> d = _solver.solver.solve(-_J.transpose() * _r);

    if (_solver.solver.info() != Eigen::Success)
        TINYAD_ERROR_throw("Linear solve failed.");

    TINYAD_ASSERT_FINITE_MAT(d);
    return d;
}

/**
 * Compute update vector d such that x + d performs a Gauss-Newton step
 * on a sum-of-squares function f(x) = sum_i r_i(x)^2.
 * Input:
 *      _r: vector of residuals
 *      _J: Jacobian of residuals. Size n_elements-by-n_vars
 */
template <typename PassiveT>
Eigen::VectorX<PassiveT> gauss_newton_direction(
        const Eigen::VectorX<PassiveT>& _r,
        const Eigen::SparseMatrix<PassiveT>& _J,
        const PassiveT& _w_identity = 0.0)
{
    LinearSolver<PassiveT> solver;
    return gauss_newton_direction(_r, _J, solver, _w_identity);
}

}
