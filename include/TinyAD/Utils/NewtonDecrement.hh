/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

/**
 * Computes (one half of the squared) Newton-Decrement.
 * The returned value is the difference between the current objective
 * value f(x) and the minimum of the quadratic approximation f(x + d).
 * It can be used as stopping criterion (newton_decrement(d, g) < eps)
 * and is affinely invariant (i.e. the same for f(x) and f(Ax + b)).
 */
template <typename PassiveT, int d>
double newton_decrement(
        const Eigen::Vector<PassiveT, d>& _d,
        const Eigen::Vector<PassiveT, d>& _g)
{
    return -0.5 * _d.dot(_g);
}

}
