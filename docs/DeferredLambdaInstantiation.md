# Deferred Lambda Instantiation in TinyAD

## Overview

TinyAD now supports deferred lambda instantiation, which significantly reduces compilation times by only instantiating evaluation variants when they are actually needed, rather than instantiating all variants upfront.

## Background

In the original implementation, when `add_elements` is called, the user-provided lambda function is immediately instantiated for all three evaluation variants:
- `PassiveEvalElementFunction` for value-only evaluation
- `ActiveFirstOrderEvalElementFunction` for gradient evaluation
- `ActiveSecondOrderEvalElementFunction` for Hessian evaluation

This happens regardless of which evaluation methods will be used later, leading to unnecessary template instantiations and increased compilation times.

## Optimization

The optimization defers lambda instantiation until the specific variant is actually needed:

1. The original lambda is stored in a type-erased container
2. Each evaluation variant is only instantiated when the corresponding evaluation method is called
3. Once instantiated, the variant is cached for future use

## Implementation Details

### Type Erasure

The implementation uses a type-erasure pattern to store the original lambda:

```cpp
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

    // Similar implementations for active_first_order and active_second_order
    // ...
};
```

### Lazy Instantiation

Each evaluation method checks if the corresponding function object has been instantiated:

```cpp
// Lazy instantiation of passive evaluation function
if (!eval_element_passive)
{
    std::lock_guard<std::mutex> lock(instantiation_mutex);
    if (!eval_element_passive) // Double-check after acquiring lock
    {
        eval_element_passive = std::make_shared<PassiveEvalElementFunction>(stored_lambda->get_passive());
    }
}
```

### Thread Safety

The implementation uses a mutex with double-checked locking to ensure thread safety during lazy instantiation.

## Benefits

### Compilation Time

The optimization significantly reduces compilation time for applications that only use a subset of evaluation methods. The reduction is particularly noticeable in large codebases with many element types and objective terms.

### Binary Size

Fewer template instantiations result in smaller binaries, which can be important for embedded systems or applications with size constraints.

### Runtime Performance

The optimization introduces a small runtime overhead for the first call to each evaluation method due to lazy instantiation. Subsequent calls have the same performance as the original implementation.

## Usage

No changes to user code are required. The optimization is transparent to users of the library.

## Example

```cpp
// Create a scalar function
auto func = TinyAD::scalar_function<2>(TinyAD::range(1));

// Add elements with a lambda function
func.template add_elements<1>(TinyAD::range(1), [](auto& element) {
    auto x = element.variables(0);
    return x[0] * x[0] + x[1] * x[1];
});

// Only the passive variant is instantiated
double f = func.eval(x);

// Now the first-order variant is instantiated
auto [f_g, g] = func.eval_with_gradient(x);

// Now the second-order variant is instantiated
auto [f_h, g_h, H] = func.eval_with_derivatives(x);
```

## Testing

The optimization has been tested for:
- Correctness of results
- Move semantics
- Thread safety

All tests pass, ensuring that the optimization maintains the same behavior as the original implementation.