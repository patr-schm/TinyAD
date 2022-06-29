/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>

namespace TinyAD
{

/*
 * TinyAD::ScalarFunction and TinyAD::VectorFunction operate on (vertex, edge, face, ...) handle types
 * of different mesh libraries. Internally, TinyAD needs to convert a set of handles to a contiguous
 * list of integers. To add support for a new mesh library, add an overload of idx_from_handle()
 * that extracts an index from a handle.
 */

/**
 * Enable support for integer handle types by overloading idx_from_handle(...).
 *
 * Allows usage:
 *    std::vector<int> variable_handles { 0, 1, 2, 3 };
 *    auto func = TinyAD::scalar_function<2>(variable_handles);
 *
 * Or alternatively:
 *    auto func = TinyAD::scalar_function<2>(TinyAD::range(4)); // Initializes handles 0, 1, 2, 3
 */
inline Eigen::Index idx_from_handle(Eigen::Index _idx)
{
    return _idx;
}

/**
 * Fallback, if no specialized overload exists.
 */
inline Eigen::Index idx_from_handle(...) // Variadic argument has lowest priorty in overload resolution
{
    TINYAD_ERROR_throw(
                "Handle type not supported. Please overload idx_from_handle() for your handle type or include one of the provided header files, e.g. TinyAD/Support/OpenMesh.hh.");
}

}
