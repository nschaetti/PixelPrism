# Math Statistics

Statistical operators are available in `pixelprism.math.functional.stats`.

## Covariance and Correlation

- `cov(x, y=None, rowvar=False, bias=False, ddof=None, dtype=...)`
- `corr(x, y=None, rowvar=False, dtype=...)`

Both support vectors or feature matrices and return scalar/matrix outputs as needed.

```python
import numpy as np
import pixelprism.math as pm
import pixelprism.math.functional.stats as S

values = pm.const(
    "values",
    data=np.array([[1.0, 2.0], [2.0, 3.5], [3.0, 5.0]], dtype=np.float32),
    dtype=pm.DType.R,
)

print(S.cov(values, rowvar=False, ddof=1).eval().value)
print(S.corr(values, rowvar=False).eval().value)
```

## Z-Score Normalization

- `zscore(x, axis=None, ddof=0, eps=1e-8, dtype=None)`

`ddof` and `eps` can be scalar expressions evaluated at runtime.

```python
import pixelprism.math as pm
import pixelprism.math.functional.stats as S

x = pm.var("x", dtype=pm.DType.R, shape=(3, 3))
ddof = pm.var("ddof", dtype=pm.DType.Z, shape=())
z = S.zscore(x, axis=0, ddof=ddof)
```

## Related Builders

Sampling builders (`normal`, `uniform`, etc.) live in the same stats functional
module and can be combined with `cov/corr/zscore` for simulation workflows.
