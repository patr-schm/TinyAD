/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/SparseCholesky>

namespace TinyAD
{

template <
        typename PassiveT = double,
        typename SolverT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<PassiveT>>>
struct LinearSolver
{
    SolverT solver;
    bool sparsity_pattern_dirty = true;
};

}
