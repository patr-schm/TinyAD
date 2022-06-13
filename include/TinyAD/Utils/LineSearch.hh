/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

template <typename PassiveT, int d>
bool armijo_condition(
        const PassiveT _f_curr,
        const PassiveT _f_new,
        const PassiveT _s,
        const Eigen::Vector<PassiveT, d>& _d,
        const Eigen::Vector<PassiveT, d>& _g,
        const PassiveT _armijo_const)
{
    return _f_new <= _f_curr + _armijo_const * _s * _d.dot(_g);
}

template <typename PassiveT, int d, typename EvalFunctionT>
Eigen::Vector<PassiveT, d> line_search(
        const Eigen::Vector<PassiveT, d>& _x0,
        const Eigen::Vector<PassiveT, d>& _d,
        const PassiveT _f,
        const Eigen::Vector<PassiveT, d>& _g,
        const EvalFunctionT& _eval, // Callable of type T(const Eigen::Vector<T, d>&)
        const PassiveT _s_max = 1.0, // Initial step size
        const PassiveT _shrink = 0.8,
        const int _max_iters = 64,
        const PassiveT _armijo_const = 1e-4)
{
    // Check input
    TINYAD_ASSERT_EQ(_x0.size(), _g.size());
    if (_s_max <= 0.0)
        TINYAD_ERROR_throw("Max step size not positive.");

    // Also try a step size of 1.0 (if valid)
    const bool try_one = _s_max > 1.0;

    Eigen::Vector<PassiveT, d> x_new = _x0;
    PassiveT s = _s_max;
    for (int i = 0; i < _max_iters; ++i)
    {
        x_new = _x0 + s * _d;
        const PassiveT f_new = _eval(x_new);
        TINYAD_ASSERT_EQ(f_new, f_new);
        if (armijo_condition(_f, f_new, s, _d, _g, _armijo_const))
            return x_new;

        if (try_one && s > 1.0 && s * _shrink < 1.0)
            s = 1.0;
        else
            s *= _shrink;
    }

    TINYAD_WARNING("Line search couldn't find improvement. Gradient max norm is " << _g.cwiseAbs().maxCoeff());

    return _x0;
}

}
