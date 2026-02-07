

import numpy as np
import pixelprism.math as pm
import pixelprism.math.render as render
import pixelprism.math.functional as F
from pixelprism.math.functional.helpers import apply_operator


data = [1, -2, 3, 0.5]
tensor_data = pm.Tensor.from_list(data, dtype=pm.DType.R)
operand_expr = pm.const(
    name=f"median_const",
    data=tensor_data.value.copy(),
    dtype=tensor_data.dtype
)
expr = F.median(operand_expr)
operand_values = tensor_data.value
expected = np.array(np.median(np.array(data)), dtype=operand_values.dtype)
print(render.to_latex(expr))
print(f"values: {expr.eval()}, expected: {expected}")

# Transpose example
matrix_expr = pm.const(
    name="transpose_matrix",
    data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    dtype=pm.DType.R
)
transpose_expr = apply_operator("transpose", (matrix_expr,), "transpose(matrix)")
matrix_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
transpose_expected = np.transpose(matrix_np)

print(render.to_latex(transpose_expr))
print(f"transpose: {transpose_expr.eval().value}, expected: {transpose_expected}")

# Determinant example
det_matrix_expr = pm.const(
    name="det_matrix",
    data=[[2.0, 3.0], [1.5, -0.5]],
    dtype=pm.DType.R
)
det_expr = apply_operator("det", (det_matrix_expr,), "det(matrix)")
det_expected = np.linalg.det(np.asarray([[2.0, 3.0], [1.5, -0.5]], dtype=np.float64))

print(render.to_latex(det_expr))
print(f"determinant: {det_expr.eval()}, expected: {det_expected}")

# Inverse example
inv_matrix_expr = pm.const(
    name="inv_matrix",
    data=[[1.0, 2.0], [3.0, 4.0]],
    dtype=pm.DType.R
)
inverse_expr = apply_operator("inverse", (inv_matrix_expr,), "inverse(matrix)")
inv_expected = np.linalg.inv(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

print(render.to_latex(inverse_expr))
print(f"inverse: {inverse_expr.eval().value}, expected: {inv_expected}")
