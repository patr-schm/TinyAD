#pragma once

#include <Eigen/Core>

// Supply vector typedefs that do not exist in old Eigen versions.
// Allows writing e.g. Eigen::Vector<T, 3> or Eigen::Vector3<T>.
// Based on https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/Matrix.h

namespace Eigen
{

#define EIGEN_MAKE_TYPEDEFS(Size, SizeSuffix)                     \
/** \ingroup matrixtypedefs */                                    \
/** \brief \cpp11 */                                              \
template <typename Type>                                          \
using Matrix##SizeSuffix = Matrix<Type, Size, Size>;              \
/** \ingroup matrixtypedefs */                                    \
/** \brief \cpp11 */                                              \
template <typename Type>                                          \
using Vector##SizeSuffix = Matrix<Type, Size, 1>;                 \
/** \ingroup matrixtypedefs */                                    \
/** \brief \cpp11 */                                              \
template <typename Type>                                          \
using RowVector##SizeSuffix = Matrix<Type, 1, Size>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Size)                           \
/** \ingroup matrixtypedefs */                                    \
/** \brief \cpp11 */                                              \
template <typename Type>                                          \
using Matrix##Size##X = Matrix<Type, Size, Dynamic>;              \
/** \ingroup matrixtypedefs */                                    \
/** \brief \cpp11 */                                              \
template <typename Type>                                          \
using Matrix##X##Size = Matrix<Type, Dynamic, Size>;

EIGEN_MAKE_TYPEDEFS(2, 2)
EIGEN_MAKE_TYPEDEFS(3, 3)
EIGEN_MAKE_TYPEDEFS(4, 4)
EIGEN_MAKE_TYPEDEFS(Dynamic, X)
EIGEN_MAKE_FIXED_TYPEDEFS(2)
EIGEN_MAKE_FIXED_TYPEDEFS(3)
EIGEN_MAKE_FIXED_TYPEDEFS(4)

/** \ingroup matrixtypedefs
  * \brief \cpp11 */
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;

/** \ingroup matrixtypedefs
  * \brief \cpp11 */
template <typename Type, int Size>
using RowVector = Matrix<Type, 1, Size>;

#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

}

