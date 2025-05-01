#pragma once

#include <TinyAD/Support/Common.hh>
#include <pmp/surface_mesh.h>

#ifdef TINYAD_ScalarFunction_DEFINED
    #error Please include this file before ScalarFunction.hh
#endif
#ifdef TINYAD_VectorFunction_DEFINED
    #error Please include this file before VectorFunction.hh
#endif

namespace TinyAD
{

/**
 * Enable support for PMP handle types by overloading idx_from_handle(...).
 *
 * Allows usage:
 *    pmp::Mesh mesh;
 *     auto func = TinyAD::scalar_function<2>(mesh.vertices());
 */
inline Eigen::Index idx_from_handle(const pmp::Handle& _h)
{
    return _h.idx();
}

}
