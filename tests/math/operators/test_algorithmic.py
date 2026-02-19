import numpy as np

import pixelprism.math as pm
import pixelprism.math.functional.algorithmic as FA


def test_algorithmic_operator_executes_registered_python_function():
    def column_sum(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        return np.sum(x, axis=1) * scale
    # end def column_sum

    FA.register_algorithm("test_column_sum", column_sum)

    x = pm.const(
        "algo_x",
        data=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        dtype=pm.DType.R,
    )
    expr = FA.algorithm(
        "test_column_sum",
        x,
        out_shape=(2,),
        out_dtype=pm.DType.R,
        params={"scale": 0.5},
    )

    value = expr.eval().value
    expected = np.array([3.0, 7.5], dtype=np.float32)
    np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
# end test_algorithmic_operator_executes_registered_python_function
