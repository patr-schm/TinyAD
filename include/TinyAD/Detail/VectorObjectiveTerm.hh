/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/SparseCore>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Detail/Element.hh>
#include <TinyAD/Detail/EvalSettings.hh>
#include <TinyAD/Utils/HessianProjection.hh>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace TinyAD
{

/**
 * Abstract base type for objective terms.
 * We need this to be able to store multiple different
 * objective term instantiations in a vector in VectorFunction.
 */
template <typename PassiveT>
struct VectorObjectiveTermBase
{
    virtual ~VectorObjectiveTermBase() = default;

    virtual Eigen::VectorX<PassiveT> eval(
            const Eigen::VectorX<PassiveT>& _x) const = 0;

    virtual void eval_with_jacobian_add(
            const Eigen::VectorX<PassiveT>& _x,
            Eigen::VectorX<PassiveT>& _r,
            std::vector<Eigen::Triplet<PassiveT>>& _J_triplets) const = 0;

    virtual void eval_with_derivatives_add(
            const Eigen::VectorX<PassiveT>& _x,
            Eigen::VectorX<PassiveT>& _r,
            std::vector<Eigen::Triplet<PassiveT>>& _J_triplets,
            std::vector<std::vector<Eigen::Triplet<PassiveT>>>& _H_triplets) const = 0;

    virtual PassiveT eval_sum_of_squares(
            const Eigen::VectorX<PassiveT>& _x) const = 0;

    virtual Eigen::Index n_outputs() const = 0;
};

/**
 * Objective term stored in VectorFunction.
 */
template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename VariableHandleT, typename ElementHandleT>
struct VectorObjectiveTerm : VectorObjectiveTermBase<PassiveT>
{
    static constexpr int n_element = variable_dimension * element_valence;

    // Scalar types. Either passive (e.g. double), or active (TinyAD::Scalar).
    // Declare separate types for first-order-only and for second-order use cases.
    using PassiveScalarType = PassiveT;
    using ActiveFirstOrderScalarType = TinyAD::Scalar<n_element, PassiveT, false>;
    using ActiveSecondOrderScalarType = TinyAD::Scalar<n_element, PassiveT, true>;

    // Element types. These are passed as argument to the user-provided lambda function.
    using PassiveElementType = Element<variable_dimension, element_valence, outputs_per_element, PassiveT, PassiveT, VariableHandleT, ElementHandleT, false>;
    using ActiveFirstOrderElementType = Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ActiveFirstOrderScalarType, VariableHandleT, ElementHandleT, true>;
    using ActiveSecondOrderElementType = Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ActiveSecondOrderScalarType, VariableHandleT, ElementHandleT, true>;

    // Return types of the user-provided lambda function.
    using PassiveEvalElementReturnType = Eigen::Vector<PassiveScalarType, outputs_per_element>;
    using ActiveFirstOrderEvalElementReturnType = Eigen::Vector<ActiveFirstOrderScalarType, outputs_per_element>;
    using ActiveSecondOrderEvalElementReturnType = Eigen::Vector<ActiveSecondOrderScalarType, outputs_per_element>;

    // Types of the user-provided lambda function.
    using PassiveEvalElementFunction = std::function<PassiveEvalElementReturnType(PassiveElementType&)>;
    using ActiveFirstOrderEvalElementFunction = std::function<ActiveFirstOrderEvalElementReturnType(ActiveFirstOrderElementType&)>;
    using ActiveSecondOrderEvalElementFunction = std::function<ActiveSecondOrderEvalElementReturnType(ActiveSecondOrderElementType&)>;

    template <typename EvalElementFunction>
    VectorObjectiveTerm(
            const std::vector<ElementHandleT>& _element_handles,
            EvalElementFunction _eval_element,
            const Eigen::Index _n_global,
            const EvalSettings& _settings)
        : n_vars_global(_n_global),
          element_handles(_element_handles),
          settings(_settings)
    {
        TINYAD_ASSERT_G(element_handles.size(), 0);

        static_assert (std::is_same_v<
                std::decay_t<decltype((_eval_element(std::declval<PassiveElementType&>())))>,
                PassiveEvalElementReturnType>,
                "Please make sure that the user-provided lambda function has the signature (const auto& element) -> TINYAD_VECTOR_TYPE(element)");

        // Instantiate _eval_element() for passive and active scalar types
        eval_element_passive = _eval_element;
        eval_element_active_first_order = _eval_element;
        eval_element_active_second_order = _eval_element;
    }

    Eigen::VectorX<PassiveT> eval(
            const Eigen::VectorX<PassiveT>& _x) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Eval elements using plain double type
        Eigen::VectorX<PassiveT> result(n_outputs());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            // Call user code and write into segment of result vector
            PassiveElementType element(element_handles[i_element], _x);
            const Eigen::Index start_idx = outputs_per_element * i_element;
            result.segment(start_idx, outputs_per_element) = eval_element_passive(element);
        }

        return result;
    }

    void eval_with_jacobian_add(
            const Eigen::VectorX<PassiveT>& _x,
            Eigen::VectorX<PassiveT>& _r,
            std::vector<Eigen::Triplet<PassiveT>>& _J_triplets) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Eval elements using active scalar type (first order)
        std::vector<ActiveFirstOrderElementType> elements(element_handles.size());
        std::vector<ActiveFirstOrderEvalElementReturnType> element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (int i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            elements[i_element] = ActiveFirstOrderElementType(element_handles[i_element], _x);

            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            element_results[i_element] = eval_element_active_first_order(elements[i_element]);

            // Assert that derivatives are finite
            for (Eigen::Index i_residual = 0; i_residual < element_results[i_element].size(); ++i_residual)
                TINYAD_ASSERT_FINITE_MAT(element_results[i_element][i_residual].grad);
        }

        // Count number of residuals and grow r
        Eigen::Index n_residuals = 0;
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
            n_residuals += element_results[i_element].size();

        const Eigen::Index residuals_start_idx = _r.size();
        _r.conservativeResize(_r.size() + n_residuals);

        // Add to global f, r, and J
        Eigen::Index i_residual = 0;
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            const auto& element = elements[i_element];
            const auto& residuals = element_results[i_element];

            for (Eigen::Index i_r_element = 0; i_r_element < residuals.size(); ++i_r_element)
            {
                const auto& residual = residuals[i_r_element];

                // Add to global residuals vector
                _r[residuals_start_idx + i_residual] = residual.val;

                // Add to global Jacobian
                for (Eigen::Index i_var_element = 0; i_var_element < element.idx_local_to_global.size(); ++i_var_element)
                {
                    _J_triplets.push_back(Eigen::Triplet<PassiveT>(
                                residuals_start_idx + i_residual,
                                element.idx_local_to_global[i_var_element],
                                residual.grad(i_var_element)));
                }

                ++i_residual;
            }
        }

        TINYAD_ASSERT_EQ(i_residual, n_residuals);
    }

    void eval_with_derivatives_add(
                const Eigen::VectorX<PassiveT>& _x,
                Eigen::VectorX<PassiveT>& _r,
                std::vector<Eigen::Triplet<PassiveT>>& _J_triplets,
                std::vector<std::vector<Eigen::Triplet<PassiveT>>>& _H_triplets) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Eval elements using active scalar type (second order)
        std::vector<ActiveSecondOrderElementType> elements(element_handles.size());
        std::vector<ActiveSecondOrderEvalElementReturnType> element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (int i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            elements[i_element] = ActiveSecondOrderElementType(element_handles[i_element], _x);

            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            element_results[i_element] = eval_element_active_second_order(elements[i_element]);

            // Assert that derivatives are finite
            for (Eigen::Index i_residual = 0; i_residual < element_results[i_element].size(); ++i_residual)
                TINYAD_ASSERT_FINITE_MAT(element_results[i_element][i_residual].grad);
        }

        // Count number of residuals and grow r and H
        Eigen::Index n_residuals = 0;
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
            n_residuals += element_results[i_element].size();

        TINYAD_ASSERT_EQ(_r.size(), _H_triplets.size());
        const Eigen::Index residuals_start_idx = _r.size();
        _r.conservativeResize(_r.size() + n_residuals);
        _H_triplets.resize(residuals_start_idx + n_residuals);

        // Add to global f, r, J, and H
        Eigen::Index i_residual = 0;
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            const ActiveSecondOrderElementType& element = elements[i_element];
            const ActiveSecondOrderEvalElementReturnType& residuals = element_results[i_element];

            for (Eigen::Index i_r_element = 0; i_r_element < residuals.size(); ++i_r_element)
            {
                const auto& residual = residuals[i_r_element];

                // Add to global residuals vector
                _r[residuals_start_idx + i_residual] = residual.val;

                // Add to global Jacobian
                for (Eigen::Index i_var_element = 0; i_var_element < element.idx_local_to_global.size(); ++i_var_element)
                {
                    _J_triplets.push_back(Eigen::Triplet<PassiveT>(
                                residuals_start_idx + i_residual,
                                element.idx_local_to_global[i_var_element],
                                residual.grad(i_var_element)));
                }

                // Add to global Hessian
                for (Eigen::Index i = 0; i < element.idx_local_to_global.size(); ++i)
                {
                    for (Eigen::Index j = 0; j < element.idx_local_to_global.size(); ++j)
                    {
                        _H_triplets[residuals_start_idx + i_residual].push_back(Eigen::Triplet<PassiveT>(
                                   element.idx_local_to_global[i],
                                   element.idx_local_to_global[j],
                                   residual.Hess(i, j)));
                    }
                }

                ++i_residual;
            }
        }

        TINYAD_ASSERT_EQ(_H_triplets.size(), _r.size());
        TINYAD_ASSERT_EQ(i_residual, n_residuals);
    }

    PassiveT eval_sum_of_squares(
            const Eigen::VectorX<PassiveT>& _x) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Eval elements using plain double type
        Eigen::VectorX<PassiveT> squared_element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
        {
            // Call user code and square results
            PassiveElementType element(element_handles[i_element], _x);
            PassiveEvalElementReturnType res = eval_element_passive(element);
            squared_element_results[i_element] = res.dot(res);
        }

        // Add up squared results
        PassiveT result = 0.0;
        for (Eigen::Index i_element = 0; i_element < element_handles.size(); ++i_element)
            result += squared_element_results[i_element];

        return result;
    }

    Eigen::Index n_outputs() const override
    {
        return outputs_per_element * element_handles.size();
    }

private:
    const Eigen::Index n_vars_global;

    const std::vector<ElementHandleT> element_handles;
    const EvalSettings& settings;

    // Instantiations of user-provided lambda
    PassiveEvalElementFunction eval_element_passive;
    ActiveFirstOrderEvalElementFunction eval_element_active_first_order;
    ActiveSecondOrderEvalElementFunction eval_element_active_second_order;
};

}
