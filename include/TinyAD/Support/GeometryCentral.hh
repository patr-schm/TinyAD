/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Support/Common.hh>
#include <geometrycentral/surface/manifold_surface_mesh.h>

#ifdef TINYAD_ScalarFunction_DEFINED
    #error Please include this file before ScalarFunction.hh
#endif
#ifdef TINYAD_VectorFunction_DEFINED
    #error Please include this file before VectorFunction.hh
#endif

namespace TinyAD
{

/**
 * Enable support for GeometryCentral handle types by overloading idx_from_handle(...).
 *
 * Allows usage:
 *     std::unique_ptr<ManifoldSurfaceMesh> mesh;
 *     auto func = TinyAD::scalar_function<2>(mesh->vertices());
 */
template <typename T, typename M>
Eigen::Index idx_from_handle(const geometrycentral::Element<T, M>& _h)
{
    return _h.getIndex();
}

}
