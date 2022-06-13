/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Support/Common.hh>
#include <polymesh/Mesh.hh>

#ifdef TINYAD_ScalarFunction_DEFINED
    #error Please include this file before ScalarFunction.hh
#endif
#ifdef TINYAD_VectorFunction_DEFINED
    #error Please include this file before VectorFunction.hh
#endif

namespace TinyAD
{

/**
 * Enable support for Polymesh handle types by overloading idx_from_handle(...).
 *
 * Allows usage:
 *     OpenMesh::TriMesh mesh;
 *     auto func = TinyAD::scalar_function<2>(mesh.vertices());
 */
template <class tag>
inline Eigen::Index idx_from_handle(const pm::primitive_handle<tag>& _h)
{
    return _h.idx.value;
}

}
