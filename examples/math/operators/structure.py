"""Flat, loop-free reproduction of the slice parametrized test for debugging."""

import numpy as np

import pixelprism.math as pm
from pixelprism.math.math_base import SliceExpr
from pixelprism.math.operators.structure import Getitem

# Static snippet lifted from tests/math/operators/test_structure.py::test_getitem_slice_combinations
TEST_SNIPPET = """expr, values = _const_vector()
slice_expr = SliceExpr.create(start=start, stop=stop, step=step)
indices = [slice_expr]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = values[slice(start, stop, step)]
"""

# Deterministic vector copied from the original helper `_const_vector`
base_values = np.arange(10, dtype=np.float32)
expr = pm.const(name="debug_getitem_vector", data=base_values.copy(), dtype=pm.DType.R)

print("Source test code:\n")
print(TEST_SNIPPET)
print("=" * 70)

# Case 1
print("Case 1")
start = 1
stop = 4
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 2
print("Case 2")
start = 1
stop = 4
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 3
print("Case 3")
start = 1
stop = 4
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 4
print("Case 4")
start = 1
stop = -1
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 5
print("Case 5")
start = 1
stop = -1
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 6
print("Case 6")
start = 1
stop = -1
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 7
print("Case 7")
start = 1
stop = None
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 8
print("Case 8")
start = 1
stop = None
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 9
print("Case 9")
start = 1
stop = None
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 10
print("Case 10")
start = -3
stop = 4
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 11
print("Case 11")
start = -3
stop = 4
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 12
print("Case 12")
start = -3
stop = 4
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 13
print("Case 13")
start = -3
stop = -1
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 14
print("Case 14")
start = -3
stop = -1
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 15
print("Case 15")
start = -3
stop = -1
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 16
print("Case 16")
start = -3
stop = None
step = 1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 17
print("Case 17")
start = -3
stop = None
step = -1
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Expected shape:", expected.shape)
print(" Actual     :", tensor.value)
print(" Actual shape :", tensor.value.shape)
print(" Inferred shape:", inferred_shape)
print("-" * 70)

# Case 18
print("Case 18")
start = -3
stop = None
step = None
indices = [SliceExpr.create(start=start, stop=stop, step=step)]
operator = Getitem(indices=indices)
tensor = operator.eval([expr])
expected = base_values[slice(start, stop, step)]
inferred_shape = operator.infer_shape([expr]).dims
print(" Input tensor:", base_values)
print(f" Slice args : start={start!r}, stop={stop!r}, step={step!r}")
print(" Expected   :", expected)
print(" Actual     :", tensor.value)
print("-" * 70)
