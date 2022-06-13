/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <Eigen/Core>

struct CustomVariableHandle
{
    CustomVariableHandle() = default;
    CustomVariableHandle(const int _idx)
        : idx(_idx) { }

    int idx = -1;
};

struct CustomElementHandle
{
    CustomElementHandle() = default;
    CustomElementHandle(const int _idx)
        : idx(_idx) { }

    int idx = -1;
};

namespace TinyAD
{

Eigen::Index idx_from_handle(const CustomVariableHandle& _vh)
{
    return _vh.idx;
}

Eigen::Index idx_from_handle(const CustomElementHandle& _eh)
{
    return _eh.idx;
}

}

// Include finite element function after custom idx_from_handle() implementation
#include <TinyAD/ScalarFunction.hh>

template <typename VariableHandleT, typename ElementHandleT>
void test_handle_types()
{
    std::vector<VariableHandleT> variable_handles;
    for (int i = 0; i < 5; ++i)
        variable_handles.push_back({ i });

    std::vector<ElementHandleT> element_handles;
    for (int i = 0; i < 10; ++i)
        element_handles.push_back({ i });

    auto func = TinyAD::scalar_function<1>(variable_handles);
    func.template add_elements<1>(element_handles, [] (auto& element)
    {
        VariableHandleT vh(TinyAD::idx_from_handle(element.handle) / 2);
        return element.variables(vh)[0];
    });

    func.eval_with_hessian_proj(Eigen::VectorXd::Zero(func.n_vars));
}

TEST(HandleTypeTest, HandleTypesInt) { test_handle_types<int, int>(); }
TEST(HandleTypeTest, HandleTypesLongInt) { test_handle_types<long int, long int>(); }
TEST(HandleTypeTest, HandleTypesCustom) { test_handle_types<CustomVariableHandle, CustomElementHandle>(); }
