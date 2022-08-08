/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Support/Common.hh>

namespace TinyAD
{

/**
 * Inside a per-element lambda function, this class provides access to variables.
 */
template <
        int variable_dimension,   // Number of scalar variables per handle.
        int element_valence,      // Number of variable handles per element.
        int outputs_per_element,  // Number of outputs produced per element.
        typename PassiveT,        // Internal scalar type. E.g. float or double.
        typename ScalarT,         // TinyAD::Scalar in active mode, PassiveT in passive mode.
        typename VariableHandleT, // Type of variable handles. E.g. int or OpenMesh::VertexHandle, ...
        typename ElementHandleT,  // Type of element handles. E.g. int or OpenMesh::FaceHandle, ...
        bool active_mode_>        // If true, derivatives are computed. ScalarType is TinyAD::Scalar. If false, ScalarType is PassiveT.
struct Element
{
    /**
     *  Number of scalar variables per element.
     */
    static constexpr int n_element = variable_dimension * element_valence;

    /**
     * Whether or not derivatives are computed in the current pass.
     * Can be accessed via TINYAD_ACTIVE_MODE(element).
     */
    static constexpr bool active_mode = active_mode_;

    /**
     * In active mode, scalars are of type TinyAD::Scalar and carry derivatives.
     * In passive mode, scalars are of type PassiveT (e.g. double).
     * Type can be accessed via TINYAD_SCALAR_TYPE(element).
     */
    using ScalarType = ScalarT;

    /**
     * Each variable handle is associated with a fixed number of scalar values
     * (e.g. 2 in planar mesh parametrization).
     * These can be either active (TinyAD::Scalar) or passive (e.g. double).
     * Type can be accessed via TINYAD_VARIABLES_TYPE(element);
     */
    using VariableVectorType = Eigen::Vector<ScalarType, variable_dimension>;

    /**
     * Each variable handle is associated with a fixed number of scalar values
     * (e.g. 2 in planar mesh parametrization).
     * This is their passive vector type, e.g., Eigen::Vector2d.
     */
    using PassiveVectorType = Eigen::Vector<PassiveT, variable_dimension>;

    /**
     * In a TinyAD::VectorFunction, each element returns several
     * residuals wrapped in a vector of this type.
     * Entries of this vector can be either active (TinyAD::Scalar) or passive (e.g. double).
     * Type can be accessed via TINYAD_VECTOR_TYPE(element);
     */
    using OutputVectorType = Eigen::Vector<ScalarType, outputs_per_element>;

    /**
     * Constructor for internal use only.
     * Instead, call func.add_elements(...).
     */
    Element() = default;

    /**
     * Elements are not copyable.
     * This prevents accidentally missing an "&" in the per-element lambda signature.
     */
    Element(const Element&) = delete;

    /**
     * Constructor for internal use only.
     * Instead, call func.add_elements(...).
     */
    Element(const ElementHandleT& _handle,
            const Eigen::VectorX<PassiveT>& _x);

    /**
     * Access the scalar variables associated with a variable handle
     * (e.g. a 2D vector in planar mesh parametrization).
     * It is logged internaly which element accesses which variables.
     * This defines a local-to-global index map and gives rise to the problem's sparsity pattern.
     */
    VariableVectorType variables(
            const VariableHandleT& _vh);

    /**
     * Access the scalar variable associated with a variable handle.
     * This function is only available if the variable dimension is 1,
     * i.e. if there is exactly one variable per handle.
     * Otherwise use element.variables() instead.
     * It is logged internaly which element accesses which variables.
     * This defines a local-to-global index map and gives rise to the problem's sparsity pattern.
     */
    ScalarType variable(
            const VariableHandleT& _vh);

    /**
     * Access the scalar variables associated with a variable handle
     * (e.g. a 2D vector in planar mesh parametrization).
     * It is not logged internaly which element accesses which variables.
     * This does not modify the local-to-global index map.
     */
    PassiveVectorType variables_passive(
            const VariableHandleT& _vh) const;

    /**
     * Access the scalar variable associated with a variable handle.
     * This function is only available if the variable dimension is 1,
     * i.e. if there is exactly one variable per handle.
     * Otherwise use element.variables_passive() instead.
     * This does not modify the local-to-global index map.
     */
    PassiveT variable_passive(
            const VariableHandleT& _vh) const;

    /**
     * Access to the element handle
     * (e.g. a triangle in mesh parametrization).
     */
    ElementHandleT handle;

    /**
     * Index map from local per-element variables (n_element many)
     * to global system indices (variable_dimension * #variable_handles many).
     */
    std::vector<Eigen::Index> idx_local_to_global;

    /**
     * Pointer to the global variable vector.
     * Do not access this directly. For internal reference only.
     * Instead, use element.variables(...).
     */
    const Eigen::VectorX<PassiveT>* x = nullptr;
};

/**
 * Element implementation:
 */

namespace
{

/**
 * Compute index in variable vector from variable handle.
 * Uses idx_from_handle(..) function.
 * You can provide additional overloads for different mesh libararies
 * (see e.g. TinyAD/Support/OpenMesh.hh).
 */
template <int variable_dimension, typename VariableHandleT>
Eigen::Index global_idx(
        const VariableHandleT& _v,
        const Eigen::Index _i_offset,
        const Eigen::Index _n_global)
{
    const Eigen::Index idx_global = variable_dimension * idx_from_handle(_v) + _i_offset;

    TINYAD_ASSERT_GEQ(idx_global, 0);
    TINYAD_ASSERT_L(idx_global, _n_global);
    return idx_global;
}

/**
  * Assert that indices obtained from variable handles (e.g. via v.idx())
  * are compact, i.e. a permutation of 0..n-1.
  */
template <typename VariableHandle>
bool variable_indices_compact(
        const std::vector<VariableHandle>& _variable_handles)
{
    std::vector<bool> idx_used(_variable_handles.size(), false);
    for (auto& v : _variable_handles)
    {
        const Eigen::Index idx = idx_from_handle(v);

        if (idx >= _variable_handles.size())
            return false; // Index not in 0..n-1
        if (idx_used[idx])
            return false; // Index occurred twice

        idx_used[idx] = true;
    }

    return true;
}

}

template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename ScalarT, typename VariableHandleT, typename ElementHandleT, bool active_mode>
Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ScalarT, VariableHandleT, ElementHandleT, active_mode>::
Element(const ElementHandleT& _handle,
        const Eigen::VectorX<PassiveT>& _x)
    : handle(_handle), x(&_x)
{
    if constexpr (active_mode)
        idx_local_to_global.reserve(n_element);
}

template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename ScalarT, typename VariableHandleT, typename ElementHandleT, bool active_mode>
auto
Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ScalarT, VariableHandleT, ElementHandleT, active_mode>::
variables(
        const VariableHandleT& _vh) -> Element::VariableVectorType
{
    // Variables associated with handle vh occupy a segment (of length variable_dimension) in x.
    // Get the start index of this segment.
    const Eigen::Index idx_global_start = global_idx<variable_dimension>(_vh, 0, x->size());
    TINYAD_ASSERT_LEQ(idx_global_start + variable_dimension, x->size());

    // Fill index map. (Local per-element indices to global indices in x).

    // First, check if this variable handle is already part of the index map (linear search).
    Eigen::Index idx_local_start = -1;
    for (Eigen::Index i = 0; i < idx_local_to_global.size(); ++i)
    {
        if (idx_local_to_global[i] == idx_global_start)
        {
            idx_local_start = i;
            TINYAD_ASSERT_EQ(idx_local_start % variable_dimension, 0);
            break;
        }
    }

    // If not already present, add all variables associated with this handle to the index map.
    if (idx_local_start == -1)
    {
        TINYAD_ASSERT_GEQ(idx_local_to_global.size(), 0);
        TINYAD_ASSERT_EQ(idx_local_to_global.size() % variable_dimension, 0);
        if (idx_local_to_global.size() >= n_element)
            TINYAD_ERROR_throw("Too many variables requested via element.variables(...).");

        idx_local_start = idx_local_to_global.size();
        for (Eigen::Index i = 0; i < variable_dimension; ++i)
            idx_local_to_global.push_back(idx_global_start + i);
    }
    TINYAD_ASSERT_LEQ(idx_local_to_global.size(), n_element);

    if constexpr (active_mode)
    {
        // Active mode: Initialize active (TinyAD::Scalar) variables.
        VariableVectorType v;
        for (Eigen::Index i = 0; i < variable_dimension; ++i)
            v[i] = ScalarType((*x)[idx_global_start + i], idx_local_start + i);

        return v;
    }
    else
    {
        // Passive mode: Just return segment of x vector.
        return x->segment(idx_global_start, variable_dimension);
    }
}

template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename ScalarT, typename VariableHandleT, typename ElementHandleT, bool active_mode>
auto
Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ScalarT, VariableHandleT, ElementHandleT, active_mode>::
variable(
        const VariableHandleT& _vh) -> Element::ScalarType
{
    static_assert (variable_dimension == 1, "element.variable(vh) is only available if variable dimension is 1. Use element.variables(vh) instead.");
    return variables(_vh)[0];
}

template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename ScalarT, typename VariableHandleT, typename ElementHandleT, bool active_mode>
auto
Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ScalarT, VariableHandleT, ElementHandleT, active_mode>::
variables_passive(
        const VariableHandleT& _vh) const -> Element::PassiveVectorType
{
    // Variables associated with handle vh occupy a segment (of length variable_dimension) in x.
    // Get the start index of this segment.
    const Eigen::Index idx_global_start = global_idx<variable_dimension>(_vh, 0, x->size());
    TINYAD_ASSERT_LEQ(idx_global_start + variable_dimension, x->size());

    // Passive mode: Just return segment of x vector.
    return x->segment(idx_global_start, variable_dimension);
}

template <int variable_dimension, int element_valence, int outputs_per_element, typename PassiveT, typename ScalarT, typename VariableHandleT, typename ElementHandleT, bool active_mode>
auto
Element<variable_dimension, element_valence, outputs_per_element, PassiveT, ScalarT, VariableHandleT, ElementHandleT, active_mode>::
variable_passive(
        const VariableHandleT& _vh) const -> PassiveT
{
    static_assert (variable_dimension == 1, "element.variable_passive(vh) is only available if variable dimension is 1. Use element.variables_passive(vh) instead.");
    return variables_passive(_vh)[0];
}

/**
  * Convenience macro to check whether the current eval pass is active or passive.
  */
#define TINYAD_ACTIVE_MODE(element) std::decay_t<decltype(element)>::active_mode

/**
 * Convenience macro to infer the current scalar type from an element.
 * (e.g. TinyAD::Scalar or double).
 * Use this to declare the return type of per-element lambda functions, e.g.:
 *
 *     func.add_elements<1>(..., [] (auto& element) -> TINYAD_SCALAR_TYPE(element) { ... });
 *
 * Or to declare variables depending on the scalar type, e.g.:
 *
 *     using T = TINYAD_SCALAR_TYPE(element);
 *     Eigen::Vector<T, 2> v = element.variables(...);
 */
#define TINYAD_SCALAR_TYPE(element) typename std::decay_t<decltype(element)>::ScalarType

/**
 * Convenience macro for the type returned by element.variables(...), e.g.:
 *
 *     using VecT = TINYAD_VARIABLES_TYPE(element);
 *     VecT v = element.variables(...);
 */
#define TINYAD_VARIABLES_TYPE(element) typename std::decay_t<decltype(element)>::VariableVectorType

/**
 * Convenience macro for the per-element return type of vector functions.
 * Use this to declare the return type lambda functions, e.g.:
 *
 *     auto func = vector_function<...>(...);
 *     func.add_elements<1>(..., [] (auto& element) -> TINYAD_VECTOR_TYPE(element) { ... });
 */
#define TINYAD_VECTOR_TYPE(element) typename std::decay_t<decltype(element)>::OutputVectorType

}
