/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#ifndef TINYAD_ScalarFunction_DEFINED
#include <TinyAD/ScalarFunction.hh>
#endif

#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Support/Common.hh>

namespace TinyAD
{

/**
 * ScalarFunction implementation:
 */

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
ScalarFunction(ScalarFunction&& _other)
    : settings(std::move(_other.settings)),
      n_vars(_other.n_vars),
      n_elements(_other.n_elements),
      variable_handles(std::move(_other.variable_handles)),
      objective_terms(std::move(_other.objective_terms))
{

}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>&
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
operator=(ScalarFunction&& _other)
{
    settings = std::move(_other.settings);
    n_vars = _other.n_vars;
    n_elements = _other.n_elements;
    variable_handles = std::move(_other.variable_handles);
    objective_terms = std::move(_other.objective_terms);
    return *this;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
ScalarFunction(
        std::vector<VariableHandleT> _variable_handles,
        const EvalSettings& _settings)
    : settings(_settings),
      n_vars(variable_dimension * _variable_handles.size()),
      variable_handles(std::move(_variable_handles))
{
    static_assert (variable_dimension >= 1, "Variable dimension needs to be at least 1.");

    TINYAD_ASSERT(!variable_handles.empty());
    TINYAD_ASSERT(variable_indices_compact(variable_handles));
    TINYAD_ASSERT_G(n_vars, 0);
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
template <int element_valence, typename ElementHandleRangeT, typename EvalElementFunction>
void
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
add_elements(
        const ElementHandleRangeT& _element_range,
        EvalElementFunction _eval_element)
{
    static_assert (element_valence >= 0, "Element valence needs to be non-negative.");

    // If this line does not compile: Make sure to pass a range with .begin() and .end() methods.
    // E.g. you could create a range of integers via TinyAD::range(n).
    using ElementHandle = typename std::decay_t<decltype(*_element_range.begin())>;

    // Assertion fails with Polymesh because polymesh::end_iterator has no operator*
    // static_assert (std::is_same_v<ElementHandle, typename std::decay_t<decltype(*_element_range.end())>>, "Please supply a valid range (with begin() and end()) as _element_range");

    // Copy handles into vector
    std::vector<ElementHandle> element_handles;
    element_handles.reserve(count(_element_range));
    for (auto eh : _element_range)
        element_handles.push_back(eh);

    // Store objective term
    using ObjectiveType = ScalarObjectiveTerm<
                variable_dimension,
                element_valence,
                PassiveT,
                VariableHandleT,
                ElementHandle>;

    objective_terms.push_back(std::make_unique<ObjectiveType>(
                element_handles, _eval_element, n_vars, settings));

    n_elements += element_handles.size();
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
Eigen::VectorX<PassiveT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
x_from_data(
        std::function<PassiveVariableVectorType(VariableHandleT)> _read_user_data) const
{
    Eigen::VectorX<PassiveT> x = Eigen::VectorX<PassiveT>::Constant(n_vars, (PassiveT)NAN);
    for (auto v : variable_handles)
    {
        const Eigen::Vector<PassiveT, variable_dimension> user_vec = _read_user_data(v);
        for (Eigen::Index idx_local = 0; idx_local < variable_dimension; ++idx_local)
        {
            const Eigen::Index idx_global = global_idx<variable_dimension>(v, idx_local, n_vars);
            x[idx_global] = user_vec[idx_local];
        }
    }
    TINYAD_ASSERT_FINITE_MAT(x);

    return x;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
x_to_data(
        const Eigen::VectorX<PassiveT>& _x,
        std::function<void(VariableHandleT, PassiveVariableVectorType)> _write_user_data) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    for (auto v : variable_handles)
    {
        PassiveVariableVectorType vec;
        for (Eigen::Index i = 0; i < variable_dimension; ++i)
            vec[i] = _x[global_idx<variable_dimension>(v, i, n_vars)];

        _write_user_data(v, vec);
    }
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
PassiveT
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT f = 0.0;
    for (auto& objective : objective_terms)
        f += objective->eval(_x);

    return f;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
PassiveT
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
operator()(
        const Eigen::VectorX<PassiveT>& _x) const
{
    return eval(_x);
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_gradient(
        const Eigen::VectorX<PassiveT>& _x,
        PassiveT& _f,
        Eigen::VectorX<PassiveT>& _g) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    _f = 0.0;
    _g = Eigen::VectorX<PassiveT>::Zero(n_vars);

    for (auto& objective : objective_terms)
        objective->eval_with_gradient_add(_x, _f, _g);
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<PassiveT, Eigen::VectorX<PassiveT>>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_gradient(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT f = 0.0;
    Eigen::VectorX<PassiveT> g;
    eval_with_gradient(_x, f, g);

    return std::tuple<PassiveT, Eigen::VectorX<PassiveT>>(f, std::move(g));
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_derivatives(
        const Eigen::VectorX<PassiveT>& _x,
        PassiveT& _f,
        Eigen::VectorX<PassiveT>& _g,
        Eigen::SparseMatrix<PassiveT>& _H) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    _f = 0.0;
    _g = Eigen::VectorX<PassiveT>::Zero(n_vars);
    _H = Eigen::SparseMatrix<PassiveT>(n_vars, n_vars);
    std::vector<Eigen::Triplet<PassiveT>> H_triplets;

    for (auto& objective : objective_terms)
        objective->eval_with_derivatives_add(_x, _f, _g, H_triplets, false, NAN);

    _H.setFromTriplets(H_triplets.begin(), H_triplets.end());
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_derivatives(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT f = 0.0;
    Eigen::VectorX<PassiveT> g;
    Eigen::SparseMatrix<PassiveT> H;
    eval_with_derivatives(_x, f, g, H);

    return std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>(f, std::move(g), std::move(H));
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
Eigen::SparseMatrix<PassiveT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_hessian(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT f = 0.0;
    Eigen::VectorX<PassiveT> g;
    Eigen::SparseMatrix<PassiveT> H;
    eval_with_derivatives(_x, f, g, H);

    return H;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
Eigen::SparseMatrix<PassiveT>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_hessian_of_quadratic() const
{
    return eval_hessian(Eigen::VectorXd::Zero(n_vars));
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_hessian_proj(
        const Eigen::VectorX<PassiveT>& _x,
        PassiveT& _f,
        Eigen::VectorX<PassiveT>& _g,
        Eigen::SparseMatrix<PassiveT>& _H_proj,
        const PassiveT& _projection_eps) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    _f = 0.0;
    _g = Eigen::VectorX<PassiveT>::Zero(n_vars);
    _H_proj = Eigen::SparseMatrix<PassiveT>(n_vars, n_vars);
    std::vector<Eigen::Triplet<PassiveT>> H_proj_triplets;

    for (auto& objective : objective_terms)
        objective->eval_with_derivatives_add(_x, _f, _g, H_proj_triplets, true, _projection_eps);

    _H_proj.setFromTriplets(H_proj_triplets.begin(), H_proj_triplets.end());
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
ScalarFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_hessian_proj(
        const Eigen::VectorX<PassiveT>& _x,
        const PassiveT& _projection_eps) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT f = 0.0;
    Eigen::VectorX<PassiveT> g;
    Eigen::SparseMatrix<PassiveT> H_proj;
    eval_with_hessian_proj(_x, f, g, H_proj, _projection_eps);

    return std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>(f, std::move(g), std::move(H_proj));
}

template <int variable_dimension, typename PassiveT, typename VariableRangeT>
auto scalar_function(
        const VariableRangeT& _variable_range,
        const EvalSettings& _settings)
{
    // If this line does not compile: Make sure to pass a range with .begin() and .end() methods.
    // E.g. you could create a range of integers via TinyAD::range(n).
    using VariableHandle = typename std::decay_t<decltype(*_variable_range.begin())>;

    // Assertion fails with Polymesh because polymesh::end_iterator has no operator*
    // static_assert (std::is_same_v<VariableHandle, typename std::decay_t<decltype(*_variable_range.end())>>, "Please supply a valid range (with begin() and end()) as _variable_range");

    // Copy handles into vectors
    std::vector<VariableHandle> variable_handles;
    variable_handles.reserve(count(_variable_range));
    for (auto vh : _variable_range)
        variable_handles.push_back(vh);

    return ScalarFunction<variable_dimension, PassiveT, VariableHandle>(
                std::move(variable_handles), _settings);
}

}
