

import numpy as np
import pixelprism.math as pm
import pixelprism.math.render as render
import pixelprism.math.functional as F


data = [1, -2, 3, 0.5]
tensor_data = pm.Tensor.from_list(data, dtype=pm.DType.FLOAT32)
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

