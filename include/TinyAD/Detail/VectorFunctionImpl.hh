/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#ifndef TINYAD_VectorFunction_DEFINED
#include <TinyAD/VectorFunction.hh>
#endif

#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Support/Common.hh>

namespace TinyAD
{

/**
 * VectorFunction implementation:
 */

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
VectorFunction(VectorFunction&& _other)
    : settings(std::move(_other.settings)),
      n_vars(_other.n_vars),
      n_elements(_other.n_elements),
      variable_handles(std::move(_other.variable_handles)),
      objective_terms(std::move(_other.objective_terms))
{

}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>&
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
operator=(VectorFunction&& _other)
{
    settings = std::move(_other.settings);
    n_vars = _other.n_vars;
    n_elements = _other.n_elements;
    variable_handles = std::move(_other.variable_handles);
    objective_terms = std::move(_other.objective_terms);
    return *this;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
VectorFunction(
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
template <int element_valence, int outputs_per_element, typename ElementHandleRangeT, typename EvalElementFunction>
void
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
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
    using ObjectiveType = VectorObjectiveTerm<
                variable_dimension,
                element_valence,
                outputs_per_element,
                PassiveT,
                VariableHandleT,
                ElementHandle>;

    objective_terms.push_back(std::make_unique<ObjectiveType>(
                element_handles, _eval_element, n_vars, settings));

    n_elements += element_handles.size();
    n_outputs += outputs_per_element * element_handles.size();
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
Eigen::VectorX<PassiveT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
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
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
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
Eigen::VectorX<PassiveT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval(const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    Eigen::VectorX<PassiveT> result(n_outputs);
    Eigen::Index start_idx = 0;
    for (auto& objective : objective_terms)
    {
        result.segment(start_idx, objective->n_outputs()) = objective->eval(_x);
        start_idx += objective->n_outputs();
    }

    return result;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
Eigen::VectorX<PassiveT>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
operator()(
        const Eigen::VectorX<PassiveT>& _x) const
{
    return eval(_x);
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_jacobian(
        const Eigen::VectorX<PassiveT>& _x,
        Eigen::VectorX<PassiveT>& _r,
        Eigen::SparseMatrix<PassiveT>& _J) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    _r = Eigen::VectorX<PassiveT>();
    std::vector<Eigen::Triplet<PassiveT>> J_triplets;

    for (auto& objective : this->objective_terms)
        objective->eval_with_jacobian_add(_x, _r, J_triplets);

    _J = Eigen::SparseMatrix<PassiveT>(_r.size(), this->n_vars);
    _J.setFromTriplets(J_triplets.begin(), J_triplets.end());
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_jacobian(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    Eigen::VectorX<PassiveT> r;
    Eigen::SparseMatrix<PassiveT> J;
    eval_with_jacobian(_x, r, J);

    return std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>(
                std::move(r), std::move(J));
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_derivatives(
        const Eigen::VectorX<PassiveT>& _x,
        Eigen::VectorX<PassiveT>& _r,
        Eigen::SparseMatrix<PassiveT>& _J,
        std::vector<Eigen::SparseMatrix<PassiveT>>& _H) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    _r = Eigen::VectorX<PassiveT>();
    std::vector<Eigen::Triplet<PassiveT>> J_triplets;
    std::vector<std::vector<Eigen::Triplet<PassiveT>>> H_triplets;

    for (auto& objective : this->objective_terms)
        objective->eval_with_derivatives_add(_x, _r, J_triplets, H_triplets);

    _J = Eigen::SparseMatrix<PassiveT>(_r.size(), this->n_vars);
    _J.setFromTriplets(J_triplets.begin(), J_triplets.end());

    TINYAD_ASSERT_EQ(_J.rows(), n_outputs);
    TINYAD_ASSERT_EQ(H_triplets.size(), n_outputs);
    _H.resize(n_outputs);
    for (Eigen::Index i_output = 0; i_output < n_outputs; ++i_output)
    {
        _H[i_output] = Eigen::SparseMatrix<PassiveT>(this->n_vars, this->n_vars);
        _H[i_output].setFromTriplets(H_triplets[i_output].begin(), H_triplets[i_output].end());
    }
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>, std::vector<Eigen::SparseMatrix<PassiveT>>>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_with_derivatives(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    Eigen::VectorX<PassiveT> r;
    Eigen::SparseMatrix<PassiveT> J;
    std::vector<Eigen::SparseMatrix<PassiveT>> H;
    eval_with_derivatives(_x, r, J, H);

    return std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>, std::vector<Eigen::SparseMatrix<PassiveT>>>(
                std::move(r), std::move(J), std::move(H));
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
PassiveT
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_sum_of_squares(const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), n_vars);

    PassiveT result = 0.0;
    for (auto& objective : objective_terms)
        result += objective->eval_sum_of_squares(_x);

    return result;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
void
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_sum_of_squares_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            Eigen::VectorX<PassiveT>& _r,
            Eigen::SparseMatrix<PassiveT>& _J) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    eval_with_jacobian(_x, _r, _J);
    _f = _r.dot(_r);
    _g = 2.0 * _J.transpose() * _r;
}

template <int variable_dimension, typename PassiveT, typename VariableHandleT>
std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
VectorFunction<variable_dimension, PassiveT, VariableHandleT>::
eval_sum_of_squares_with_derivatives(
        const Eigen::VectorX<PassiveT>& _x) const
{
    TINYAD_ASSERT_EQ(_x.size(), this->n_vars);

    PassiveT f = 0.0;
    Eigen::VectorX<PassiveT> g;
    Eigen::VectorX<PassiveT> r;
    Eigen::SparseMatrix<PassiveT> J;
    eval_sum_of_squares_with_derivatives(_x, f, g, r, J);

    return std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>(
                f, std::move(g), std::move(r), std::move(J));
}

template <int variable_dimension, typename PassiveT, typename VariableRangeT>
auto vector_function(
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

    return VectorFunction<variable_dimension, PassiveT, VariableHandle>(
                std::move(variable_handles), _settings);
}

}
