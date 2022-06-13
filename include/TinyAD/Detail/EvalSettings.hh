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

inline int get_n_threads(
        const EvalSettings& _settings)
{
    #ifdef _OPENMP
        if (_settings.n_threads >= 1)
            return _settings.n_threads;
        else
            return std::max(1, omp_get_max_threads() - 1); // Don't use all available threads to prevent random hangs.
    #else
        return 1;
    #endif
}

}
