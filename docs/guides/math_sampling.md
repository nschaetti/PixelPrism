# Math Sampling

PixelPrism sampling operators create symbolic random tensors. Parameters can be scalar
`MathExpr`, so values are resolved at `eval()` time.

## Available Sampling Operators

- `normal(shape, loc, scale, dtype)`
- `uniform(shape, low, high, dtype)`
- `randint(shape, low, high=None, dtype)`
- `poisson(shape, lam, dtype)`
- `bernoulli(shape, p, dtype)`

## Runtime Parameter Evaluation

All distribution parameters are expressions, including literal values converted to
constants. If context values change, a new evaluation uses new parameters.

```python
import pixelprism.math as pm
import pixelprism.math.functional.stats as S

loc = pm.var("loc", dtype=pm.DType.R, shape=())
scale = pm.var("scale", dtype=pm.DType.R, shape=())
sample = S.normal(shape=(4, 4), loc=loc, scale=scale)

with pm.new_context():
    pm.set_value("loc", 2.0)
    pm.set_value("scale", 0.5)
    print(sample.eval().value)
```

## Validation Timing

Distribution constraints are checked at evaluation time:

- `normal`: `scale >= 0`
- `uniform`: `high > low`
- `randint`: valid integer bounds
- `poisson`: `lam >= 0`
- `bernoulli`: `p in [0, 1]`

This enables fully symbolic parameterization.
