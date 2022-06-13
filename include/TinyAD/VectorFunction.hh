/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <memory>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Detail/EvalSettings.hh>
#include <TinyAD/Detail/VectorObjectiveTerm.hh>

namespace TinyAD
{

/**
 * Class implementing a differentiable vector function f: R^n -> R^m,
 * defined via a set of elements. Each element produces a segment of the output vector.
 */
template <
        int variable_dimension,   // Number of scalar variables per handle. E.g. 2 in 2D mesh parametrization.
        typename PassiveT,        // Internal scalar type. E.g. float or double.
        typename VariableHandleT> // Type of variable handles. E.g. int or OpenMesh::VertexHandle, ...
struct VectorFunction
{
    using PassiveScalarType = PassiveT;
    using PassiveVariableVectorType = Eigen::Vector<PassiveScalarType, variable_dimension>;
    static constexpr bool is_vector_function = true;

    // Vector function is not copyable but movable
    VectorFunction() = default;
    VectorFunction(const VectorFunction&) = delete;
    VectorFunction(VectorFunction&& _other);
    VectorFunction& operator=(const VectorFunction&) = delete;
    VectorFunction& operator=(VectorFunction&& _other);

    /**
     * Instead of this constructor, use vector_function<..>(..),
     * which helps with deducing template arguments.
     */
    VectorFunction(
            std::vector<VariableHandleT> _variable_handles,
            const EvalSettings& _settings);

    /**
     * Add a set of elements and a lambda function evaluating each element.
     * Each element produces a segment of the output vector.
     * Can be called multiple times to add different terms.
     */
    template <
            int element_valence,          // Number of variable handles accessed per element.
            int outputs_per_element,      // Number of entries in the output vector produced by one element.
            typename ElementHandleRangeT, // Type of element handles. E.g. int or OpenMesh::Face handle, ... Deduced automatically.
            typename EvalElementFunction> // Type of per-element eval function. Deduced automatically.
    void add_elements(
            const ElementHandleRangeT& _element_range,
            EvalElementFunction _eval_element);

    /**
     * Assemble variable vector x from user data.
     * Uses internal index map from variable handles to entries of x.
     *
     * Pass a lambda function that takes a variable handle and returns its associated scalar values.
     *
     * Example:
     *      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) { return param.row(v_idx); });
     */
    Eigen::VectorX<PassiveT> x_from_data(
            std::function<PassiveVariableVectorType(VariableHandleT)> _read_user_data) const;

    /**
     * Write variable vector x to user data.
     * Uses internal index map from variable handles to entries of x.
     *
     * Pass a lambda function that takes a variable handle its associated scalar values
     * and writes these values to the user data structure.
     *
     * Example:
     *      func.x_to_data(x, [&] (int v_idx, const Eigen::Vector2d& p) { param.row(v_idx) = p; });
     */
    void x_to_data(
            const Eigen::VectorX<PassiveT>& _x,
            std::function<void(VariableHandleT, PassiveVariableVectorType)> _write_user_data) const;

    /**
     * Evaluate function without computing derivatives.
     */
    Eigen::VectorX<PassiveT> eval(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate function without computing derivatives.
     */
    Eigen::VectorX<PassiveT> operator()(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate result vector and Jacobian matrix.
     * J has dimension n_outputs-by-n_variables.
     */
    void eval_with_jacobian(
            const Eigen::VectorX<PassiveT>& _x,
            Eigen::VectorX<PassiveT>& _r,
            Eigen::SparseMatrix<PassiveT>& _J) const;

    /**
     * Evaluate result vector and Jacobian matrix.
     * J has dimension n_outputs-by-n_variables.
     */
    std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
    eval_with_jacobian(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate result vector, Jacobian matrix, and Hessian tensor.
     * J has dimension n_outputs-by-n_variables.
     * H has dimension n_output-by-n_variables-by-n_variables.
     */
    void eval_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x,
            Eigen::VectorX<PassiveT>& _r,
            Eigen::SparseMatrix<PassiveT>& _J,
            std::vector<Eigen::SparseMatrix<PassiveT>>& _H) const;

    /**
     * Evaluate result vector, Jacobian matrix, and Hessian tensor.
     * J has dimension n_outputs-by-n_variables.
     * H has dimension n_output-by-n_variables-by-n_variables.
     */
    std::tuple<Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>, std::vector<Eigen::SparseMatrix<PassiveT>>>
    eval_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate f(x) = sum_i (r_i(x))^2 without derivatives.
     */
    PassiveT eval_sum_of_squares(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate f(x) = sum_i (r_i(x))^2.
     * Returns the gradient of f.
     * Returns the vector of residuals r, of size n_elements.
     * Returns Jacobian matrix J with gradients as rows,
     * i.e. J_ij = d r_i / d x_j.
     * J has dimension n_outputs-by-n_variables.
     */
    void eval_sum_of_squares_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            Eigen::VectorX<PassiveT>& _r,
            Eigen::SparseMatrix<PassiveT>& _J) const;

    /**
     * Evaluate f(x) = sum_i (r_i(x))^2.
     * Returns the gradient of f.
     * Returns the vector of residuals r, of size n_elements.
     * Returns Jacobian matrix J with gradients as rows,
     * i.e. J_ij = d r_i / d x_j.
     * J has dimension n_outputs-by-n_variables.
     */
    std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
    eval_sum_of_squares_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Change settings before calling eval(..).
     *
     * Example:
     *      func.settings.n_threads = 4;
     */
    EvalSettings settings;

    /**
     * Number of scalar variables (size of variable vector x).
     * This is variable_dimension * #variable_handles.
     */
    Eigen::Index n_vars = 0;

    /**
      * Current number of elements
      */
    Eigen::Index n_elements = 0;

    /**
      * Current number of outputs
      */
    Eigen::Index n_outputs = 0;

    std::vector<VariableHandleT> variable_handles;
    std::vector<std::unique_ptr<VectorObjectiveTermBase<PassiveT>>> objective_terms;
};

/**
 * Use this to construct vector functions.
 * Automatically deduces variable handle type.
 */
template <
        int variable_dimension,     // Number of scalar variables per handle.
        typename PassiveT = double, // Internal scalar type. E.g. float or double.
        typename VariableRangeT>    // Range type of variable handles. E.g. std::vector<int>. Deduced automatically.
auto vector_function(
        const VariableRangeT& _variable_range,
        const EvalSettings& _settings = EvalSettings());

}

#define TINYAD_VectorFunction_DEFINED
#include <TinyAD/Detail/VectorFunctionImpl.hh>
