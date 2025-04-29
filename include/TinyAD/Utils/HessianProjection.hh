/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Eigenvalues>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

// Eigenvalues are clamped to be larger or equal to this value.
// If negative: Negative eigenvalues are replaced by their absolute value.
constexpr double default_hessian_projection_eps = 1e-9;

/**
 * Check if matrix is diagonally dominant and has positive diagonal entries.
 * This is a sufficient condition for positive-definiteness
 * and can be used as an early out to avoid eigen decomposition.
 */
template <int k, typename PassiveT>
bool positive_diagonally_dominant(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const PassiveT& _eps)
{
    for (Eigen::Index i = 0; i < _H.rows(); ++i)
    {
        PassiveT off_diag_abs_sum = 0.0;
        for(Eigen::Index j = 0; j < _H.cols(); ++j)
        {
            if (i != j)
                off_diag_abs_sum += std::abs(_H(i, j));
        }

        if (_H(i, i) < off_diag_abs_sum + _eps)
            return false;
    }

    return true;
}

/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition.
 */
template <int k, typename PassiveT>
void project_positive_definite(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const PassiveT& _eigenvalue_eps)
{
    if constexpr (k == 0)
    {
        return;
    }
    else
    {
        using MatT = Eigen::Matrix<PassiveT, k, k>;

        // Early out if sufficient condition is fulfilled
        if (positive_diagonally_dominant<k, PassiveT>(_H, _eigenvalue_eps))
            return;

        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<MatT> eig(_H);
        MatT D = eig.eigenvalues().asDiagonal();

        // Clamp all eigenvalues to eps
        bool all_positive = true;
        for (Eigen::Index i = 0; i < _H.rows(); ++i)
        {
            if (_eigenvalue_eps < 0)
            {
                // Use absolute eigenvalue strategy (https://arxiv.org/html/2406.05928v3)
                if (D(i, i) < 0)
                {
                    D(i, i) = -D(i, i);
                    all_positive = false;
                }
            }
            else
            {
                // Use clamping strategy
                if (D(i, i) < _eigenvalue_eps)
                {
                    D(i, i) = _eigenvalue_eps;
                    all_positive = false;
                }
            }
        }

        // Do nothing if all eigenvalues were already at least eps
        if (all_positive)
            return;

        // Re-assemble matrix using clamped eigenvalues
        _H = eig.eigenvectors() * D * eig.eigenvectors().transpose();
        TINYAD_ASSERT_FINITE_MAT(_H);
    }
}

}
