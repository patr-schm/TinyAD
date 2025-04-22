/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace TinyAD
{

struct EvalSettings
{
    /**
     * Number of OpenMP threads.
     * Positive number or -1 for max available threads.
     */
    int n_threads = -1;
};

}
