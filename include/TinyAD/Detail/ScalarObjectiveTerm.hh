/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/SparseCore>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Detail/Element.hh>
#include <TinyAD/Detail/Parallel.hh>
#include <TinyAD/Detail/EvalSettings.hh>
#include <TinyAD/Utils/HessianProjection.hh>
#include <mutex>
#include <memory>
#include <functional>
#include <type_traits>

namespace TinyAD
{

/**
 * Abstract base type for objective terms.
 * We need this to be able to store multiple different
 * objective term instantiations in a vector in ScalarFunction.
 */
template <typename PassiveT>
struct ScalarObjectiveTermBase
{
    virtual ~ScalarObjectiveTermBase() = default;

    virtual PassiveT eval(
            const Eigen::VectorX<PassiveT>& _x) const = 0;

    virtual void eval_with_gradient_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g) const = 0;

    virtual void eval_with_derivatives_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            std::vector<Eigen::Triplet<PassiveT>>& _H_proj_triplets,
            const bool _project_hessian,
            const PassiveT& _projection_eps) const = 0;
};

/**
 * Objective term stored in ScalarFunction.
 */
template <int variable_dimension, int element_valence, typename PassiveT, typename VariableHandleT, typename ElementHandleT>
struct ScalarObjectiveTerm : ScalarObjectiveTermBase<PassiveT>
{
    static constexpr int n_element = variable_dimension * element_valence;

    // Scalar types. Either passive (e.g. double), or active (TinyAD::Scalar).
    // Declare separate types for first-order-only and for second-order use cases.
    using PassiveScalarType = PassiveT;
    using ActiveFirstOrderScalarType = TinyAD::Scalar<n_element, PassiveT, false>;
    using ActiveSecondOrderScalarType = TinyAD::Scalar<n_element, PassiveT, true>;

    // Element types. These are passed as argument to the user-provided lambda function.
    using PassiveElementType = Element<variable_dimension, element_valence, 1, PassiveT, PassiveT, VariableHandleT, ElementHandleT, false>;
    using ActiveFirstOrderElementType = Element<variable_dimension, element_valence, 1, PassiveT, ActiveFirstOrderScalarType, VariableHandleT, ElementHandleT, true>;
    using ActiveSecondOrderElementType = Element<variable_dimension, element_valence, 1, PassiveT, ActiveSecondOrderScalarType, VariableHandleT, ElementHandleT, true>;

    // Return types of the user-provided lambda function.
    using PassiveEvalElementReturnType = PassiveScalarType;
    using ActiveFirstOrderEvalElementReturnType = ActiveFirstOrderScalarType;
    using ActiveSecondOrderEvalElementReturnType = ActiveSecondOrderScalarType;

    // Types of the user-provided lambda function.
    using PassiveEvalElementFunction = std::function<PassiveEvalElementReturnType(PassiveElementType&)>;
    using ActiveFirstOrderEvalElementFunction = std::function<ActiveFirstOrderEvalElementReturnType(ActiveFirstOrderElementType&)>;
    using ActiveSecondOrderEvalElementFunction = std::function<ActiveSecondOrderEvalElementReturnType(ActiveSecondOrderElementType&)>;

    // Non-template base class for storing the lambda
    struct LambdaBase
    {
        virtual ~LambdaBase() = default;
        virtual PassiveEvalElementFunction get_passive() = 0;
        virtual ActiveFirstOrderEvalElementFunction get_active_first_order() = 0;
        virtual ActiveSecondOrderEvalElementFunction get_active_second_order() = 0;
    };

    // Template implementation for specific lambda types
    template <typename F>
    struct LambdaImpl : LambdaBase
    {
        LambdaImpl(F&& f) : func(std::forward<F>(f)) {}

        PassiveEvalElementFunction get_passive() override
        {
            return [this](PassiveElementType& element) -> PassiveEvalElementReturnType {
                return func(element);
            };
        }

        ActiveFirstOrderEvalElementFunction get_active_first_order() override
        {
            return [this](ActiveFirstOrderElementType& element) -> ActiveFirstOrderEvalElementReturnType {
                return func(element);
            };
        }

        ActiveSecondOrderEvalElementFunction get_active_second_order() override
        {
            return [this](ActiveSecondOrderElementType& element) -> ActiveSecondOrderEvalElementReturnType {
                return func(element);
            };
        }

        F func;
    };

    template <typename EvalElementFunction>
    ScalarObjectiveTerm(
            const std::vector<ElementHandleT>& _element_handles,
            EvalElementFunction _eval_element,
            const Eigen::Index _n_global,
            const EvalSettings& _settings)
        : n_vars_global(_n_global),
          element_handles(_element_handles),
          settings(_settings)
    {
        static_assert (std::is_same_v<
                std::decay_t<decltype((_eval_element(std::declval<PassiveElementType&>())))>,
                PassiveEvalElementReturnType>,
                "Please make sure that the user-provided lambda function has the signature (const auto& element) -> TINYAD_SCALAR_TYPE(element)");

        // Store the original lambda for deferred instantiation
        stored_lambda = std::make_unique<LambdaImpl<EvalElementFunction>>(std::forward<EvalElementFunction>(_eval_element));
    }

    // Move constructor
    ScalarObjectiveTerm(ScalarObjectiveTerm&& other) noexcept
        : n_vars_global(other.n_vars_global),
          element_handles(std::move(other.element_handles)),
          settings(other.settings),
          stored_lambda(std::move(other.stored_lambda)),
          eval_element_passive(std::move(other.eval_element_passive)),
          eval_element_active_first_order(std::move(other.eval_element_active_first_order)),
          eval_element_active_second_order(std::move(other.eval_element_active_second_order))
    {
    }

    // Move assignment
    ScalarObjectiveTerm& operator=(ScalarObjectiveTerm&& other) noexcept
    {
        if (this != &other)
        {
            // n_vars_global and settings are const, so we can't move them
            element_handles = std::move(other.element_handles);
            stored_lambda = std::move(other.stored_lambda);
            eval_element_passive = std::move(other.eval_element_passive);
            eval_element_active_first_order = std::move(other.eval_element_active_first_order);
            eval_element_active_second_order = std::move(other.eval_element_active_second_order);
        }
        return *this;
    }

    PassiveT eval(
            const Eigen::VectorX<PassiveT>& _x) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Lazy instantiation of passive evaluation function
        if (!eval_element_passive)
        {
            std::lock_guard<std::mutex> lock(instantiation_mutex);
            if (!eval_element_passive) // Double-check after acquiring lock
            {
                eval_element_passive = std::make_shared<PassiveEvalElementFunction>(stored_lambda->get_passive());
            }
        }

        // Eval elements using plain double type
        std::vector<PassiveT> element_results(element_handles.size());

        parallel_for(element_handles.size(), settings, [&] (Eigen::Index i_element)
        {
            // Call user code
            PassiveElementType element(element_handles[i_element], _x);
            element_results[i_element] = (*eval_element_passive)(element);
        });

        // Sum up results
        PassiveT f = 0.0;
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_results.size(); ++i_element)
            f += element_results[i_element];

        return f;
    }

    void eval_with_gradient_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);
        TINYAD_ASSERT_EQ(_g.size(), n_vars_global);

        // Lazy instantiation of first-order evaluation function
        if (!eval_element_active_first_order)
        {
            std::lock_guard<std::mutex> lock(instantiation_mutex);
            if (!eval_element_active_first_order) // Double-check after acquiring lock
            {
                eval_element_active_first_order = std::make_shared<ActiveFirstOrderEvalElementFunction>(
                    stored_lambda->get_active_first_order());
            }
        }

        // Eval elements using active scalar type
        std::vector<ActiveFirstOrderElementType> elements(element_handles.size());
        std::vector<ActiveFirstOrderScalarType> element_results(element_handles.size());

        parallel_for(element_handles.size(), settings, [&] (Eigen::Index i_element)
        {
            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            elements[i_element] = ActiveFirstOrderElementType(element_handles[i_element], _x);
            element_results[i_element] = (*eval_element_active_first_order)(elements[i_element]);

            // Assert that derivatives are finite
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].grad);
        });

        // Add to global f and g
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            _f += element_results[i_element].val;

            // Add to global gradient
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
                _g[elements[i_element].idx_local_to_global[i]] += element_results[i_element].grad[i];
        }
    }

    void eval_with_derivatives_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            std::vector<Eigen::Triplet<PassiveT>>& _H_triplets,
            const bool _project_hessian,
            const PassiveT& _projection_eps) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);
        TINYAD_ASSERT_EQ(_g.size(), n_vars_global);

        // Lazy instantiation of second-order evaluation function
        if (!eval_element_active_second_order)
        {
            std::lock_guard<std::mutex> lock(instantiation_mutex);
            if (!eval_element_active_second_order) // Double-check after acquiring lock
            {
                eval_element_active_second_order = std::make_shared<ActiveSecondOrderEvalElementFunction>(
                    stored_lambda->get_active_second_order());
            }
        }

        // Eval elements using active scalar type
        std::vector<ActiveSecondOrderElementType> elements(element_handles.size());
        std::vector<ActiveSecondOrderScalarType> element_results(element_handles.size());

        parallel_for(element_handles.size(), settings, [&] (Eigen::Index i_element)
        {
            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            elements[i_element] = ActiveSecondOrderElementType(element_handles[i_element], _x);
            element_results[i_element] = (*eval_element_active_second_order)(elements[i_element]);

            if (_project_hessian)
                project_positive_definite<n_element, PassiveT>(element_results[i_element].Hess, _projection_eps);

            // Assert that derivatives are finite
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].grad);
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].Hess);
        });

        // Add to global f, g and H
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            _f += element_results[i_element].val;

            // Add to global gradient
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
                _g[elements[i_element].idx_local_to_global[i]] += element_results[i_element].grad[i];

            // Add to global Hessian
            using SparseIndex = typename Eigen::SparseMatrix<PassiveT>::StorageIndex;
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
            {
                for (Eigen::Index j = 0; j < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++j)
                {
                    _H_triplets.push_back(Eigen::Triplet<PassiveT>(
                               (SparseIndex)elements[i_element].idx_local_to_global[i],
                               (SparseIndex)elements[i_element].idx_local_to_global[j],
                               element_results[i_element].Hess(i, j)));
                }
            }
        }
    }

private:
    const Eigen::Index n_vars_global;

    const std::vector<ElementHandleT> element_handles;
    const EvalSettings& settings;

    // Storage for the original lambda
    std::unique_ptr<LambdaBase> stored_lambda;

    // Mutex for thread-safe lazy instantiation
    mutable std::mutex instantiation_mutex;

    // Lazy-instantiated function objects
    mutable std::shared_ptr<PassiveEvalElementFunction> eval_element_passive;
    mutable std::shared_ptr<ActiveFirstOrderEvalElementFunction> eval_element_active_first_order;
    mutable std::shared_ptr<ActiveSecondOrderEvalElementFunction> eval_element_active_second_order;
};

}
