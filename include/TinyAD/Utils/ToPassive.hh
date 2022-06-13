/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD
{

// Include this file for a fallback no-op version of to_passive(...)
// without needing to include Scalar.hh

template <typename PassiveT>
const PassiveT& to_passive(const PassiveT& a)
{
    return a;
}

}

// Additional passive-type functions for which Scalar.hh
// offers active overloads:

template <typename PassiveT>
const PassiveT sqr(const PassiveT& a)
{
    return a * a;
}
