/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <memory>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Detail/EvalSettings.hh>
#include <TinyAD/Detail/ScalarObjectiveTerm.hh>
#include <TinyAD/Utils/HessianProjection.hh>

namespace TinyAD
{

/**
 * Class implementing a differentiable scalar function f: R^n -> R,
 * defined as a sum of element functions.
 *
 * Example (2D mesh parametrization).
 * 2 variables per vertex. Objective per triangle, using 3 vertices each:
 *
 *      auto func = scalar_function<2>(TinyAD::range(V.rows()));
 *      func.add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
 *      {
 *          using T = TINYAD_SCALAR_TYPE(element);
 *          int f_idx = element.handle;
 *          Eigen::Vector2<T> a = element.variables(F(f_idx, 0));
 *          Eigen::Vector2<T> b = element.variables(F(f_idx, 1));
 *          Eigen::Vector2<T> c = element.variables(F(f_idx, 2));
 *          return ...;
 *      });
 *      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) { return ...; });
 *      auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
 */
template <
        int variable_dimension,   // Number of scalar variables per handle. E.g. 2 in 2D mesh parametrization.
        typename PassiveT,        // Internal scalar type. E.g. float or double.
        typename VariableHandleT> // Type of variable handles. E.g. int or OpenMesh::VertexHandle, ...
struct ScalarFunction
{
    using PassiveScalarType = PassiveT;
    using PassiveVariableVectorType = Eigen::Vector<PassiveScalarType, variable_dimension>;
    static constexpr bool is_vector_function = false;

    // Scalar function is not copyable but movable
    ScalarFunction() = default;
    ScalarFunction(const ScalarFunction&) = delete;
    ScalarFunction(ScalarFunction&& _other);
    ScalarFunction& operator=(const ScalarFunction&) = delete;
    ScalarFunction& operator=(ScalarFunction&& _other);

    /**
     * Instead of this contructor, use scalar_function<..>(..),
     * which helps with deducing template arguments.
     */
    ScalarFunction(
            std::vector<VariableHandleT> _variable_handles,
            const EvalSettings& _settings);

    /**
     * Add a set of elements (summands in the objective)
     * and a lambda function that evaluates each element.
     * Can be called multiple times to add different terms.
     * 
     * This is the **static** overload, where every element has the same valence,
     * i.e. accesses the same number of variable handles.
     */
    template <
            int... ElementValences,       // A single integer: The number of variable handles accessed per element.
            typename ElementHandleRangeT, // Type of element handles. E.g. int or OpenMesh::Face handle or other. Deduced automatically.
            typename EvalElementFunction> // Type of per-element eval function. Deduced automatically.
    std::enable_if_t<(sizeof...(ElementValences) == 1)>
    add_elements(
            const ElementHandleRangeT& _element_range,
            EvalElementFunction _eval_element);

    /**
     * Add a set of elements (summands in the objective)
     * and a lambda function that evaluates each element.
     * Can be called multiple times to add different terms.
     *
     * This is the **"dynamic"** overload, where each element can access a different number of variable handles at run time.
     * To still benefit from compile-time optimizations, this will internally instantiate the static version
     * for a couple of different valences. This list of valences has to be provided as template arguments,
     * e.g. add_elements<3, 4, 5>(...). Each run-time valence will be mappted to the exact or next higher static valence.
     * The highest run-time valence may not exceed the highest static valence.
     */
    template <
            int... ElementValences,       // List of common element valences. E.g. 5, 6, 7, 16 for meshes with max valence 16.
            typename ElementHandleRangeT, // Type of element handles. E.g. int or OpenMesh::Face handle or other. Deduced automatically.
            typename EvalElementFunction> // Type of per-element eval function. Deduced automatically.
    std::enable_if_t<(sizeof...(ElementValences) >= 2)>
    add_elements(
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
    PassiveT eval(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate function without computing derivatives.
     */
    PassiveT operator()(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate function with gradient.
     */
    void eval_with_gradient(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g) const;

    /**
     * Evaluate function with gradient.
     */
    std::tuple<PassiveT, Eigen::VectorX<PassiveT>>
    eval_with_gradient(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate function with gradient and Hessian.
     */
    void eval_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            Eigen::SparseMatrix<PassiveT>& _H) const;

    /**
     * Evaluate function with gradient and Hessian.
     */
    std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
    eval_with_derivatives(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate Hessian matrix only.
     */
    Eigen::SparseMatrix<PassiveT> eval_hessian(
            const Eigen::VectorX<PassiveT>& _x) const;

    /**
     * Evaluate Hessian assuming function is quadratic, i.e. Hessian is independent of x.
     * Warning: It is not checked if the function is actually quadratic.
     */
    Eigen::SparseMatrix<PassiveT> eval_hessian_of_quadratic() const;

    /**
     * Evaluate function with gradient and Hessian.
     * The returned Hessian matrix is positive-definite (via per-element projection).
     */
    void eval_with_hessian_proj(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            Eigen::SparseMatrix<PassiveT>& _H_proj,
            const PassiveT& _projection_eps = default_hessian_projection_eps) const;

    /**
     * Evaluate function with gradient and Hessian.
     * The returned Hessian matrix is positive-definite (via per-element projection).
     */
    std::tuple<PassiveT, Eigen::VectorX<PassiveT>, Eigen::SparseMatrix<PassiveT>>
    eval_with_hessian_proj(
            const Eigen::VectorX<PassiveT>& _x,
            const PassiveT& _projection_eps = default_hessian_projection_eps) const;

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

    std::vector<VariableHandleT> variable_handles;
    std::vector<std::unique_ptr<ScalarObjectiveTermBase<PassiveT>>> objective_terms;
};

/**
 * Use this to construct scalar functions.
 * Automatically deduces variable handle type.
 */
template <
        int variable_dimension,     // Number of scalar variables per handle.
        typename PassiveT = double, // Internal scalar type. E.g. float or double.
        typename VariableRangeT>    // Range type of variable handles. E.g. std::vector<int>. Deduced automatically.
auto scalar_function(
        const VariableRangeT& _variable_range,
        const EvalSettings& _settings = EvalSettings());

}

#define TINYAD_ScalarFunction_DEFINED
#include <TinyAD/Detail/ScalarFunctionImpl.hh>
