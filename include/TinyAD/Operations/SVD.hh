/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>

namespace TinyAD
{

template <typename T>
int sign(const T& _x)
{
    if (_x < T(0.0))
        return  -1;
    else if (_x > T(0.0))
        return 1;
    else
        return 0;
}

/**
 * 2x2 closed-form SVD.
 */
template <typename T>
void svd(
        const Eigen::Matrix2<T>& _A,
        Eigen::Matrix2<T>& _U,
        Eigen::Vector2<T>& _S,
        Eigen::Matrix2<T>& _V)
{
    // From https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
    // [U,SIG,V] = svd2x2(A) finds the SVD of 2x2 matrix A
    // where U and V are orthogonal, SIG is diagonal,
    // and A=U*SIG*V’
    // Find U such that U*A*A’*U’=diag
    Eigen::Matrix2<T> Su = _A * _A.transpose();
    T phi = 0.5 * atan2(Su(0, 1) + Su(1, 0), Su(0, 0) - Su(1, 1));
    T Cphi = cos(phi);
    T Sphi = sin(phi);
    _U << Cphi, -Sphi,
          Sphi, Cphi;

    // Find W such that W’*A’*A*W=diag
    Eigen::Matrix2<T> Sw = _A.transpose() * _A;
    T theta = 0.5 * atan2(Sw(0, 1) + Sw(1, 0), Sw(0, 0) - Sw(1, 1));
    T Ctheta = cos(theta);
    T Stheta = sin(theta);
    Eigen::Matrix2<T> W;
    W << Ctheta, -Stheta,
         Stheta, Ctheta;

    // Find the singular values from U
    T SUsum = Su(0, 0) + Su(1, 1);
    T SUdif = sqrt(sqr(Su(0, 0) - Su(1, 1)) + 4.0 * Su(0, 1) * Su(1, 0));
    _S << sqrt((SUsum + SUdif) / 2.0), sqrt((SUsum - SUdif) / 2.0);

    // Find the correction matrix for the right side
    Eigen::Matrix2<T> S = _U.transpose() * _A * W;
    Eigen::Vector2<T> C(sign(S(0, 0)), sign(S(1, 1)));
    _V = W * C.asDiagonal();
}

/**
 * Compute closest orthogonal 2x2 matrix via SVD.
 * Returns U * V^T for SVD A = U * S * V^T.
 */
template <typename T>
Eigen::Matrix2<T> closest_orthogonal(
        const Eigen::Matrix2<T>& _A)
{
    // Based on https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/

    // Find U such that U*A*A’*U’=diag
    Eigen::Matrix2<T> Su = _A * _A.transpose();
    T phi = 0.5 * atan2(Su(0, 1) + Su(1, 0), Su(0, 0) - Su(1, 1));
    T Cphi = cos(phi);
    T Sphi = sin(phi);
    Eigen::Matrix2<T> U;
    U << Cphi, -Sphi,
         Sphi, Cphi;

    // Find W such that W’*A’*A*W=diag
    Eigen::Matrix2<T> Sw = _A.transpose() * _A;
    T theta = 0.5 * atan2(Sw(0, 1) + Sw(1, 0), Sw(0, 0) - Sw(1, 1));
    T Ctheta = cos(theta);
    T Stheta = sin(theta);
    Eigen::Matrix2<T> W;
    W << Ctheta, -Stheta,
         Stheta, Ctheta;

    // Find the correction matrix for the right side
    Eigen::Matrix2<T> S = U.transpose() * _A * W;
    Eigen::Vector2<T> C(sign(S(0, 0)), sign(S(1, 1)));
    Eigen::Matrix2<T> V = W * C.asDiagonal();

    return U * V.transpose();
}

}
