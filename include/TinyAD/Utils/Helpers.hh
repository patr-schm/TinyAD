/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <TinyAD/Utils/Out.hh>

namespace TinyAD
{

/**
 * Create vector of indices from 0 to n-1.
 */
inline std::vector<Eigen::Index> range(
        const Eigen::Index _n)
{
    TINYAD_ASSERT_GEQ(_n, 0);

    std::vector<Eigen::Index> r(_n);
    for (Eigen::Index i = 0; i < _n; ++i)
        r[i] = i;

    return r;
}

/**
 * Count elements in range.
 * (This exists because std::distance cannot handle
 * different iterator types for begin and end.)
 */
template <typename RangeT>
Eigen::Index count(
        const RangeT& _range)
{
    Eigen::Index n = 0;
    for (const auto& r : _range)
        ++n;

    return n;
}

/**
 * Assemble matrix from column vectors.
 */
template <typename Derived>
auto col_mat(
        const Eigen::MatrixBase<Derived>& _v0,
        const Eigen::MatrixBase<Derived>& _v1)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 2 * Derived::ColsAtCompileTime> M;

    M << _v0, _v1;

    return M;
}

/**
 * Assemble matrix from column vectors.
 */
template <typename Derived>
auto col_mat(
        const Eigen::MatrixBase<Derived>& _v0,
        const Eigen::MatrixBase<Derived>& _v1,
        const Eigen::MatrixBase<Derived>& _v2)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 3 * Derived::ColsAtCompileTime> M;

    M << _v0, _v1, _v2;

    return M;
}

/**
 * Sparse identity matrix.
 */
template <typename PassiveT>
Eigen::SparseMatrix<PassiveT> identity(
        const Eigen::Index _n)
{
    Eigen::SparseMatrix<PassiveT> Id(_n, _n);
    Id.setIdentity();

    return Id;
}

}
