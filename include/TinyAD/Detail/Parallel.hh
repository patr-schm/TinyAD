/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Detail/EvalSettings.hh>
#include <Eigen/Dense>
#include <optional>
#include <atomic>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace TinyAD
{

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

/**
 * Runs the specified lambda function in an omp for loop.
 * On throw: Catches first exception and re-throws it in the main thread.
 */
inline void parallel_for(
        const Eigen::Index n,
        const EvalSettings& settings,
        std::function<void(Eigen::Index)> function_body)
{
    std::mutex exception_mutex;
    std::optional<std::exception_ptr> exception_ptr;
    std::atomic<bool> cancel_requested(false);

    #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (cancel_requested)
            continue;

        try
        {
            function_body(i);
        }
        catch (const std::exception& ex)
        {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!exception_ptr.has_value())
                exception_ptr = std::make_exception_ptr(ex);

            cancel_requested = true;
        }
    }

    // Re-throw, if we stored an exception
    if (exception_ptr.has_value())
        std::rethrow_exception(exception_ptr.value());
}

}
