/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <cmath>
#include <Eigen/Dense>

#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/ToPassive.hh>

namespace TinyAD
{

/**
  * Forward-differentiable scalar type with constructors for passive and active variables.
  * Each scalar carries its gradient and Hessian w.r.t. a variable vector.
  *     k: Size of variable vector at compile time. (Eigen::Dynamic is possible, but experimental and slow.)
  *     PassiveT: Internal floating point type, e.g. double.
  *     with_hessian: Set to false for gradient-only mode.
  *     hess_row_start, hess_col_start, hess_rows, hess_cols: Restrict Hessian computation to a sub-block
  */
template <int k, typename PassiveT, bool with_hessian = true, int hess_row_start = 0, int hess_col_start = 0, int hess_rows = k, int hess_cols = k>
struct Scalar
{
    // Make template arguments available as members
    static constexpr int k_ = k;
    static constexpr bool with_hessian_ = with_hessian;
    static constexpr bool dynamic_mode_ = k == Eigen::Dynamic;
    static constexpr bool truncated_hessian_ = hess_row_start != 0 || hess_col_start != 0 || hess_rows != k || hess_cols != k;

    // Validate template arguments
    static_assert (k_ >= 0 || k_ == Eigen::Dynamic, "Variable dimension k has to be non-negative or Eigen::Dynamic.");
    static_assert (!truncated_hessian_ || with_hessian_, "with_hessian needs to be true when restricting Hessian to a sub-block.");
    static_assert (!(dynamic_mode_ && truncated_hessian_), "Truncated Hessian is not supported in dynamic mode.");
    static_assert (!truncated_hessian_ || (hess_row_start >= 0 && hess_rows >= 0 && hess_row_start + hess_rows <= k), "Selected Hessian block has to be <= k-by-k");
    static_assert (!truncated_hessian_ || (hess_col_start >= 0 && hess_cols >= 0 && hess_col_start + hess_cols <= k), "Selected Hessian block has to be <= k-by-k");

    // Determine derivative data types at compile time.
    // Use 0-by-0 if no Hessian required.
    // Hessian might be truncated to a (rectangular) block smaller than k-by-k.
    using GradType = Eigen::Matrix<PassiveT, k, 1>;
    using HessType = typename std::conditional_t<
                with_hessian,
                Eigen::Matrix<PassiveT, hess_rows, hess_cols>,
                Eigen::Matrix<PassiveT, 0, 0>>;

    // ///////////////////////////////////////////////////////////////////////////
    // TinyAD::Scalar constructors
    // ///////////////////////////////////////////////////////////////////////////

    /// Default constructor, copy, move, assignment
    Scalar() = default;
    Scalar(const Scalar& _rhs) = default;
    Scalar(Scalar&& _rhs) = default;
    Scalar& operator=(const Scalar& _rhs) = default;
    Scalar& operator=(Scalar&& _rhs) = default;

    /// Passive variable a.k.a. constant.
    /// Gradient and Hessian are zero.
    Scalar(PassiveT _val)
        : val(_val)
    {
        static_assert(!dynamic_mode_, "Implicit constructor is only available in static mode. Either choose k at runtime or use make_passive(val, k_dynamic).");
    }

    /// Active variable.
    ///     _idx: index in variable vector
    Scalar(PassiveT _val, Eigen::Index _idx)
        : val(_val)
    {
        static_assert(!dynamic_mode_, "This constructor is only available in static mode. Either choose k at compile time or use make_active(val, idx, k_dynamic).");

        TINYAD_ASSERT_GEQ(_idx, 0);
        TINYAD_ASSERT_L(_idx, k);
        grad(_idx) = 1.0;
    }

    /// Initialize scalar with known derivatives
    static Scalar known_derivatives(PassiveT _val, const GradType& _grad, const HessType& _Hess)
    {
        Scalar res;
        res.val = _val;
        res.grad = _grad;

        if constexpr (with_hessian)
            res.Hess = _Hess;

        return res;
    }

    /// Initialize scalar with known derivatives (univariate case)
    static Scalar known_derivatives(PassiveT _val, PassiveT _grad, PassiveT _Hess)
    {
        static_assert(k == 1 || dynamic_mode_, "Constructor only available for univariate case. Call overload with vector-valued arguments.");

        Scalar res;
        res.val = _val;
        res.grad = GradType::Constant(1, _grad);

        if constexpr (with_hessian)
            res.Hess = HessType::Constant(1, 1, _Hess);

        return res;
    }

    /// Initialize passive variable a.k.a. constant with zero derivatives of size _k_dynamic.
    /// Only necessary in dynamic mode to pass derivative size at run time.
    /// In static mode, use the Scalar(val) constructor instead.
    static Scalar make_passive(PassiveT _val, Eigen::Index _k_dynamic)
    {
        if constexpr (!dynamic_mode_)
            return Scalar(_val);
        else
        {
            Scalar res;
            res.val = _val;
            res.grad = GradType::Zero(_k_dynamic);

            if constexpr (with_hessian)
                res.Hess = HessType::Zero(_k_dynamic, _k_dynamic);

            return res;
        }
    }

    /// Initialize active variable with derivatives of size _k_dynamic.
    /// Only necessary in dynamic mode to pass derivative size at run time.
    /// In static mode, use the Scalar(val, idx) constructor instead.
    static Scalar make_active(PassiveT _val, Eigen::Index _idx, Eigen::Index _k_dynamic)
    {
        if constexpr (!dynamic_mode_)
            return Scalar(_val, _idx);
        else
        {
            TINYAD_ASSERT_L(_idx, _k_dynamic);

            Scalar res;
            res.val = _val;
            res.grad = GradType::Zero(_k_dynamic);
            res.grad[_idx] = 1.0;

            if constexpr (with_hessian)
                res.Hess = HessType::Zero(_k_dynamic, _k_dynamic);

            return res;
        }
    }

    /// Initialize an active variable vector of size k from given values.
    static Eigen::Matrix<Scalar, k, 1> make_active(
            const Eigen::Matrix<PassiveT, Eigen::Dynamic, 1>& _passive)
    {
        if constexpr (dynamic_mode_)
        {
            const Eigen::Index k_dynamic = _passive.size();
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> active(k_dynamic);
            for (Eigen::Index i = 0; i < k_dynamic; ++i)
                active[i] = Scalar::make_active(_passive[i], i, k_dynamic);

            return active;
        }
        else
        {
            TINYAD_ASSERT_EQ(_passive.size(), k);
            Eigen::Matrix<Scalar, k, 1> active(k);
            for (Eigen::Index i = 0; i < k; ++i)
                active[i] = Scalar(_passive[i], i);

            return active;
        }
    }

    /// Initialize an active variable vector of size k from given values.
    static Eigen::Matrix<Scalar, k, 1> make_active(
            std::initializer_list<PassiveT> _passive)
    {
        return make_active(Eigen::Map<const Eigen::Matrix<PassiveT, Eigen::Dynamic, 1>>(_passive.begin(), _passive.size()));
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Convenience functions
    // ///////////////////////////////////////////////////////////////////////////

    /// Compute outer product grad_a * grad_b^T, restricted to a specific block.
    /// The result is a (possibly rectangular) matrix of size less or equal than k-by-k.
    static auto outer(
            const GradType& grad_a, // size k
            const GradType& grad_b) // size k
    {
        if constexpr (truncated_hessian_)
            return grad_a.template segment<hess_rows>(hess_row_start) * grad_b.template segment<hess_cols>(hess_col_start).transpose();
        else
            return grad_a * grad_b.transpose(); // This line is here to not break dynamic mode
    }

    /// Apply chain rule to compute f(a(x)) and its derivatives.
    static Scalar chain(
            const PassiveT& val,  // f
            const PassiveT& grad, // df/da
            const PassiveT& Hess, // ddf/daa
            const Scalar& a)
    {
        Scalar res;
        res.val = val;
        res.grad = grad * a.grad;

        if constexpr (with_hessian)
            res.Hess = Hess * outer(a.grad, a.grad) + grad * a.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Unary operators
    // ///////////////////////////////////////////////////////////////////////////

    friend Scalar operator-(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res;
        res.val = -a.val;
        res.grad = -a.grad;

        if constexpr(with_hessian)
            res.Hess = -a.Hess;

        return res;
    }

    friend Scalar sqrt(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT f = std::sqrt(a.val);
        return chain(
                    f,
                    (PassiveT)0.5 / f,
                    (PassiveT)-0.25 / (f * a.val),
                    a);
    }

    friend Scalar sqr(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res;
        res.val = a.val * a.val;
        res.grad = 2.0 * a.val * a.grad;

        if constexpr (with_hessian)
            res.Hess = 2.0 * (a.val * a.Hess + outer(a.grad, a.grad)); // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar pow(
            const Scalar& a,
            const int& e)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT f2 = (PassiveT)std::pow(a.val, e - 2);
        const PassiveT f1 = f2 * a.val;
        const PassiveT f = f1 * a.val;

        return chain(
                    f,
                    e * f1,
                    e * (e - 1) * f2,
                    a);
    }

    friend Scalar pow(
            const Scalar& a,
            const PassiveT& e)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT f2 = std::pow(a.val, e - (PassiveT)2.0);
        const PassiveT f1 = f2 * a.val;
        const PassiveT f = f1 * a.val;

        return chain(
                    f,
                    e * f1,
                    e * (e - (PassiveT)1.0) * f2,
                    a);
    }

    friend Scalar fabs(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if (a.val >= 0.0)
            return chain(a.val, 1.0, 0.0, a);
        else
            return chain(-a.val, -1.0, 0.0, a);
    }

    friend Scalar abs(
            const Scalar& a)
    {
        return fabs(a);
    }

    friend Scalar exp(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT exp_a = std::exp(a.val);
        return chain(exp_a, exp_a, exp_a, a);
    }

    friend Scalar log(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT a_inv = (PassiveT)1.0 / a.val;
        return chain(
                    std::log(a.val),
                    a_inv,
                    -a_inv / a.val,
                    a);
    }

    friend Scalar log2(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT a_inv = (PassiveT)1.0 / a.val / (PassiveT)std::log(2.0);
        return chain(
                    std::log2(a.val),
                    a_inv,
                    -a_inv / a.val,
                    a);
    }

    friend Scalar log10(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT a_inv = (PassiveT)1.0 / a.val / (PassiveT)std::log(10.0);
        return chain(
                    std::log10(a.val),
                    a_inv,
                    -a_inv / a.val,
                    a);
    }

    friend Scalar sin(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT sin_a = std::sin(a.val);
        return chain(
                    sin_a,
                    std::cos(a.val),
                    -sin_a,
                    a);
    }

    friend Scalar cos(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT cos_a = std::cos(a.val);
        return chain(
                    cos_a,
                    -std::sin(a.val),
                    -cos_a,
                    a);
    }

    friend Scalar tan(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT cos = std::cos(a.val);
        const PassiveT cos_2 = cos * cos;
        const PassiveT cos_3 = cos_2 * cos;
        return chain(
                    std::tan(a.val),
                    (PassiveT)1.0 / cos_2,
                    (PassiveT)2.0 * std::sin(a.val) / cos_3,
                    a);
    }

    friend Scalar asin(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT s = (PassiveT)1.0 - a.val * a.val;
        const PassiveT s_sqrt = std::sqrt(s);
        return chain(
                    std::asin(a.val),
                    (PassiveT)1.0 / s_sqrt,
                    a.val / s_sqrt / s,
                    a);
    }

    friend Scalar acos(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_ASSERT_G(a.val, -1.0);
        TINYAD_ASSERT_L(a.val, 1.0);

        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT s = (PassiveT)1.0 - a.val * a.val;
        const PassiveT s_sqrt = std::sqrt(s);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(s);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(s_sqrt);
        TINYAD_ASSERT(s > 0.0);

        return chain(
                    std::acos(a.val),
                    (PassiveT)-1.0 / s_sqrt,
                    -a.val / s_sqrt / s,
                    a);
    }

    friend Scalar atan(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT s = a.val * a.val + (PassiveT)1.0;
        return chain(
                    std::atan(a.val),
                    (PassiveT)1.0 / s,
                    (PassiveT)-2.0 * a.val / s / s,
                    a);
    }

    friend Scalar sinh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT sinh_a = std::sinh(a.val);
        return chain(
                    sinh_a,
                    std::cosh(a.val),
                    sinh_a,
                    a);
    }

    friend Scalar cosh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT cosh_a = std::cosh(a.val);
        return chain(
                    cosh_a,
                    std::sinh(a.val),
                    cosh_a,
                    a);
    }

    friend Scalar tanh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT cosh = std::cosh(a.val);
        const PassiveT cosh_2 = cosh * cosh;
        const PassiveT cosh_3 = cosh_2 * cosh;
        return chain(
                    std::tanh(a.val),
                    (PassiveT)1.0 / cosh_2,
                    (PassiveT)-2.0 * std::sinh(a.val) / cosh_3,
                    a);
    }

    friend Scalar asinh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT s = a.val * a.val + (PassiveT)1.0;
        const PassiveT s_sqrt = std::sqrt(s);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(s);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(s_sqrt);

        return chain(
                    std::asinh(a.val),
                    (PassiveT)1.0 / s_sqrt,
                    -a.val / s_sqrt / s,
                    a);
    }

    friend Scalar acosh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT sm = a.val - (PassiveT)1.0;
        const PassiveT sp = a.val + (PassiveT)1.0;
        const PassiveT sm_sqrt = std::sqrt(sm);
        const PassiveT sp_sqrt = std::sqrt(sp);
        const PassiveT prod = sm_sqrt * sp_sqrt;
        TINYAD_CHECK_FINITE_IF_ENABLED_d(sm_sqrt);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(sp_sqrt);

        return chain(
                    std::acosh(a.val),
                    (PassiveT)1.0 / prod,
                    -a.val / prod / sm / sp,
                    a);
    }

    friend Scalar atanh(
            const Scalar& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        const PassiveT s = (PassiveT)1.0 - a.val * a.val;
        return chain(
                    std::atanh(a.val),
                    (PassiveT)1.0 / s,
                    (PassiveT)2.0 * a.val / s / s,
                    a);
    }

    friend bool isnan(
            const Scalar& a)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return std::isnan(a.val);
    }

    friend bool isinf(
            const Scalar& a)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return std::isinf(a.val);
    }

    friend bool isfinite(
            const Scalar& a)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return std::isfinite(a.val);
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Binary operators
    // ///////////////////////////////////////////////////////////////////////////

    friend Scalar operator+(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(a.grad.size(), b.grad.size());

        Scalar res;
        res.val = a.val + b.val;
        res.grad = a.grad + b.grad;

        if constexpr (with_hessian)
            res.Hess = a.Hess + b.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator+(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = a;
        res.val += b;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator+(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = b;
        res.val += a;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    Scalar& operator+=(
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(this->grad.size(), b.grad.size());

        this->val += b.val;
        this->grad += b.grad;
        if constexpr (with_hessian)
            this->Hess += b.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        return *this;
    }

    Scalar& operator+=(
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        this->val += b;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        return *this;
    }

    friend Scalar operator-(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(a.grad.size(), b.grad.size());

        Scalar res;
        res.val = a.val - b.val;
        res.grad = a.grad - b.grad;

        if constexpr (with_hessian)
            res.Hess = a.Hess - b.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator-(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = a;
        res.val -= b;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator-(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res;
        res.val = a - b.val;
        res.grad = -b.grad;

        if constexpr (with_hessian)
            res.Hess = -b.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    Scalar& operator-=(
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(this->grad.size(), b.grad.size());

        this->val -= b.val;
        this->grad -= b.grad;

        if constexpr (with_hessian)
            this->Hess -= b.Hess; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        return *this;
    }

    Scalar& operator-=(
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        this->val -= b;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(*this);
        return *this;
    }

    friend Scalar operator*(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(a.grad.size(), b.grad.size());

        Scalar res;
        res.val = a.val * b.val;
        res.grad = b.val * a.grad + a.val * b.grad;

        // Exploiting symmetry did not yield speedup in some tests
//        if constexpr (with_hessian)
//        {
//            for(Eigen::Index j = 0; j < k; ++j)
//            {
//                for(Eigen::Index i = j; i < k; ++i)
//                {
//                    res.Hess(i, j) = b.val * a.Hess(i, j) + a.grad[i] * b.grad[j] + a.grad[j] * b.grad[i] + a.val * b.Hess(i, j);
//                }
//            }
//            res.Hess = res.Hess.template selfadjointView<Eigen::Lower>();
//        }

        if constexpr (with_hessian) // Might be a Hessian block only!
            res.Hess = b.val * a.Hess + outer(a.grad, b.grad) + outer(b.grad, a.grad) + a.val * b.Hess;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator*(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_d(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = a;
        res.val *= b;
        res.grad *= b;

        if constexpr (with_hessian)
            res.Hess *= b; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator*(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_d(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = b;
        res.val *= a;
        res.grad *= a;

        if constexpr (with_hessian)
            res.Hess *= a; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    Scalar& operator*=(
            const Scalar& b)
    {
        *this = *this * b;
        return *this;
    }

    Scalar& operator*=(
            const PassiveT& b)
    {
        *this = *this * b;
        return *this;
    }

    friend Scalar operator/(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(a.grad.size(), b.grad.size());

        Scalar res;
        res.val = a.val / b.val;
        res.grad = (b.val * a.grad - a.val * b.grad) / (b.val * b.val);

        if constexpr (with_hessian) // Might be a Hessian block only!
            res.Hess = (a.Hess - outer(res.grad, b.grad) - outer(b.grad, res.grad) - res.val * b.Hess) / b.val;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator/(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res = a;
        res.val /= b;
        res.grad /= b;

        if constexpr (with_hessian)
            res.Hess /= b; // Might be a Hessian block only!

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar operator/(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        Scalar res;
        res.val = a / b.val;
        res.grad = (-a / (b.val * b.val)) * b.grad;

        if constexpr (with_hessian) // Might be a Hessian block only!
            res.Hess = (outer(-res.grad, b.grad) - outer(b.grad, res.grad) - res.val * b.Hess) / b.val;

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    Scalar& operator/=(
            const Scalar& b)
    {
        *this = *this / b;
        return *this;
    }

    Scalar& operator/=(
            const PassiveT& b)
    {
        *this = *this / b;
        return *this;
    }

    friend Scalar atan2(
            const Scalar& y,
            const Scalar& x)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(y);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(x);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(y.grad.size(), x.grad.size());

        Scalar res;
        res.val = std::atan2(y.val, x.val);

        const GradType u = x.val * y.grad - y.val * x.grad;
        const PassiveT v = x.val * x.val + y.val * y.val;
        res.grad = u / v;

        if constexpr (with_hessian)
        {
            // Might be a Hessian block only!
            const HessType du = x.val * y.Hess - y.val * x.Hess + outer(y.grad, x.grad) - outer(x.grad, y.grad);
            const GradType dv = (PassiveT)2.0 * (x.val * x.grad + y.val * y.grad);
            res.Hess = (du - res.grad * dv.transpose()) / v;
        }

        TINYAD_CHECK_FINITE_IF_ENABLED_AD(res);
        return res;
    }

    friend Scalar hypot(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        if constexpr (dynamic_mode_) TINYAD_ASSERT_EQ(a.grad.size(), b.grad.size());

        return sqrt(a * a + b * b);
    }

    friend bool operator==(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val == b.val;
    }

    friend bool operator==(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val == b;
    }

    friend bool operator==(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a == b.val;
    }

    friend bool operator!=(
            const Scalar& a,
            const Scalar& b)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val != b.val;
    }

    friend bool operator!=(
            const Scalar& a,
            const PassiveT& b)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val != b;
    }

    friend bool operator!=(
            const PassiveT& a,
            const Scalar& b)
    {
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a != b.val;
    }

    friend bool operator<(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val < b.val;
    }

    friend bool operator<(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val < b;
    }

    friend bool operator<(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a < b.val;
    }

    friend bool operator<=(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val <= b.val;
    }

    friend bool operator<=(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val <= b;
    }

    friend bool operator<=(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a <= b.val;
    }

    friend bool operator>(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val > b.val;
    }

    friend bool operator>(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val > b;
    }

    friend bool operator>(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a > b.val;
    }

    friend bool operator>=(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val >= b.val;
    }

    friend bool operator>=(
            const Scalar& a,
            const PassiveT& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a.val >= b;
    }

    friend bool operator>=(
            const PassiveT& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);
        return a >= b.val;
    }

    friend Scalar min(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return (b < a) ? b : a;
    }

    friend Scalar fmin(
            const Scalar& a,
            const Scalar& b)
    {
        return min(a, b);
    }

    friend Scalar max(
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return (a < b) ? b : a;
    }

    friend Scalar fmax(
            const Scalar& a,
            const Scalar& b)
    {
        return max(a, b);
    }

    friend Scalar clamp(
            const Scalar& x,
            const Scalar& a,
            const Scalar& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(x);
        if (x < a)
            return a;
        else if (x > b)
            return b;
        else
            return x;
    }

    // ///////////////////////////////////////////////////////////////////////////
    // std::complex operators (just spell out and differentiate the real case)
    // ///////////////////////////////////////////////////////////////////////////

    friend std::complex<Scalar> operator+(
            const std::complex<Scalar>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    friend std::complex<Scalar> operator+(
            const std::complex<PassiveT>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    friend std::complex<Scalar> operator+(
            const std::complex<Scalar>& a,
            const std::complex<PassiveT>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    friend std::complex<Scalar> operator-(
            const std::complex<Scalar>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    friend std::complex<Scalar> operator-(
            const std::complex<PassiveT>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    friend std::complex<Scalar> operator-(
            const std::complex<Scalar>& a,
            const std::complex<PassiveT>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    friend std::complex<Scalar> operator*(
            const std::complex<Scalar>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(
                    a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real());
    }

    friend std::complex<Scalar> operator*(
            const std::complex<PassiveT>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(
                    a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real());
    }

    friend std::complex<Scalar> operator*(
            const std::complex<Scalar>& a,
            const std::complex<PassiveT>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(
                    a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real());
    }

    friend std::complex<Scalar> sqr(
            const std::complex<Scalar>& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(
                    sqr(a.real()) - sqr(a.imag()),
                    2.0 * a.real() * a.imag());
    }

    friend std::complex<Scalar> operator/(
            const std::complex<Scalar>& a,
            const std::complex<Scalar>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        const Scalar denom = b.real() * b.real() + b.imag() * b.imag();
        return std::complex<Scalar>(
                    (a.real() * b.real() + a.imag() * b.imag()) / denom,
                    (a.imag() * b.real() - a.real() * b.imag()) / denom);
    }

    friend std::complex<Scalar> operator/(
            const std::complex<Scalar>& a,
            const std::complex<PassiveT>& b)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(b);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        const PassiveT denom = b.real() * b.real() + b.imag() * b.imag();
        return std::complex<Scalar>(
                    (a.real() * b.real() + a.imag() * b.imag()) / denom,
                    (a.imag() * b.real() - a.real() * b.imag()) / denom);
    }

    friend std::complex<Scalar> conj(
            const std::complex<Scalar>& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return std::complex<Scalar>(a.real(), -a.imag());
    }

    friend Scalar abs(
            const std::complex<Scalar>& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return hypot(a.real(), a.imag());
    }

    friend Scalar arg(
            const std::complex<Scalar>& a)
    {
        TINYAD_CHECK_FINITE_IF_ENABLED_AD(a);
        if constexpr (TINYAD_ENABLE_OPERATOR_LOGGING) TINYAD_DEBUG_VAR(__FUNCTION__);

        return atan2(a.imag(), a.real());
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Stream Operators
    // ///////////////////////////////////////////////////////////////////////////

    friend std::ostream& operator<<(
            std::ostream& s,
            const Scalar& a)
    {
        s << a.val << std::endl;
        s << "grad: " << a.grad << std::endl;
        if constexpr (with_hessian)
                s << "Hess: " << a.Hess;
        return s;
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Data
    // ///////////////////////////////////////////////////////////////////////////

    PassiveT val = 0.0;                // Scalar value of this (intermediate) variable.
    GradType grad = GradType::Zero(    // Gradient (first derivative) of val w.r.t. the active variable vector.
                dynamic_mode_ ? 0 : k);
    HessType Hess = HessType::Zero(    // Hessian (second derivative) of val w.r.t. the active variable vector. Might be restricted to smaller (rectangular) block.
                dynamic_mode_ ? 0 : (with_hessian ? hess_rows : 0),
                dynamic_mode_ ? 0 : (with_hessian ? hess_cols : 0));
};

// ///////////////////////////////////////////////////////////////////////////
// Overloads (Fails to build on windows otherwise)
// ///////////////////////////////////////////////////////////////////////////
template  <typename T1, typename T2>
T1 pow(const T1& a, const T2& e)
{
    return std::pow(a, e);
}

template <typename PassiveT>
PassiveT atan2(const PassiveT& _y, const PassiveT& _x)
{
    return std::atan2(_y, _x);
}

// ///////////////////////////////////////////////////////////////////////////
// Explicit conversion to passive types
// ///////////////////////////////////////////////////////////////////////////

template <int k, typename PassiveT, bool with_hessian>
PassiveT to_passive(const Scalar<k, PassiveT, with_hessian>& a) { return a.val; }

template <int k, int rows, int cols, typename ScalarT, bool with_hessian>
Eigen::Matrix<ScalarT, rows, cols> to_passive(
        const Eigen::Matrix<Scalar<k, ScalarT, with_hessian>, rows, cols>& A)
{
    Eigen::Matrix<ScalarT, rows, cols> A_passive(A.rows(), A.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            A_passive(i, j) = A(i, j).val;
    }

    return A_passive;
}

// ///////////////////////////////////////////////////////////////////////////
// TinyAD::Scalar typedefs: Float, Double, LongDouble
// ///////////////////////////////////////////////////////////////////////////

template <int k, bool with_hessian = true, int hess_row_start = 0, int hess_col_start = 0, int hess_rows = k, int hess_cols = k> using Float = Scalar<k, float, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>;
template <int k, bool with_hessian = true, int hess_row_start = 0, int hess_col_start = 0, int hess_rows = k, int hess_cols = k> using Double = Scalar<k, double, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>;
template <int k, bool with_hessian = true, int hess_row_start = 0, int hess_col_start = 0, int hess_rows = k, int hess_cols = k> using LongDouble = Scalar<k, long double, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>;

} // namespace TinyAD

// ///////////////////////////////////////////////////////////////////////////
// Eigen3 traits
// ///////////////////////////////////////////////////////////////////////////
namespace Eigen
{

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template<int k, typename PassiveT, bool with_hessian, int hess_row_start, int hess_col_start, int hess_rows, int hess_cols>
struct NumTraits<TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>>
        : NumTraits<PassiveT>
{
    typedef TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols> Real;
    typedef TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols> NonInteger;
    typedef TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols> Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = k == Eigen::Dynamic ? 1 : 1 + k + (with_hessian ? k * k : 0),
        MulCost = k == Eigen::Dynamic ? 1 : 1 + k + (with_hessian ? k * k : 0),
    };
};

/*
 * Let Eigen know that binary operations between TinyAD::Scalar and T are allowed,
 * and that the return type is TinyAD::Scalar.
 */
template<typename BinaryOp, int k, typename PassiveT, bool with_hessian, int hess_row_start, int hess_col_start, int hess_rows, int hess_cols>
struct ScalarBinaryOpTraits<TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>, PassiveT, BinaryOp>
{
    typedef TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols> ReturnType;
};

template<typename BinaryOp, int k, typename PassiveT, bool with_hessian, int hess_row_start, int hess_col_start, int hess_rows, int hess_cols>
struct ScalarBinaryOpTraits<PassiveT, TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>, BinaryOp>
{
    typedef TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols> ReturnType;
};

} // namespace Eigen
