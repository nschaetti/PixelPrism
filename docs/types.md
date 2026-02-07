## Scalar and TScalar

In **PixelPrism**, we distinguish between two types of scalars:

- **`Scalar`** — a simple wrapper around a number (`float` or `int`)
- **`TScalar`** — a symbolic or differentiable scalar used for dynamic or lazy evaluation

They are used throughout the library to allow both immediate values and computed expressions that react to changes — especially useful in animations or parameterized objects.

### Creating a Scalar

You can create a `Scalar` directly from a number:

```python
from pixelprism import Scalar

s = Scalar(3.14)
```

You can also pass an optional `on_change` callback or make the scalar `readonly`:

```python
s = Scalar(5, on_change=my_callback, readonly=True)
```

If you pass another `Scalar` as the value, its internal value will be extracted automatically.

### Creating a TScalar

A `TScalar` is defined by a function and the `Scalar` objects it depends on.

```python
from pixelprism import TScalar, Scalar

a = Scalar(5)
b = Scalar(2)

t = TScalar(lambda s1, s2: s1.value + s2.value, s1=a, s2=b)
```

This creates a `TScalar` representing `a + b`, which will update automatically when either `a` or `b` changes.

You can also provide an `on_change` callback:

```python
t = TScalar(lambda s: s.value ** 2, s=a, on_change=handle_update)
```

### ➕ Class methods to build common `TScalar` expressions

You don't need to always write lambdas. There are helper methods to simplify common operations:

#### `TScalar.add(...)`

```python
TScalar.add(scalar1, scalar2)
```

```python
t = TScalar.add(3, Scalar(2))  # returns a TScalar(5)
```

Both arguments can be numbers or `Scalar` instances.

#### `TScalar.log10(...)`

```python
TScalar.t_log10(scalar)
```

```python
a = Scalar(100)
t = TScalar.t_log10(a)  # returns a TScalar representing log10(100)
```

### Operator rules (`__add__`)

When using `+` with a `Scalar`, different result types are returned depending on the right-hand operand:

| Right-hand type (`other`) | Result type   | Description                                                              |
| ------------------------- | ------------- | ------------------------------------------------------------------------ |
| `float` / `int`           | `Scalar`      | `Scalar(self.value + other)`                                             |
| `Scalar`                  | `Scalar`      | `Scalar(self.value + other.value)`                                       |
| `TScalar`                 | `TScalar`     | `TScalar(lambda s, o: s.value + o.value, s=self, o=other)`               |
| `Point2D`                 | `Point2D`     | `Point2D(self.value + other.x, self.value + other.y)`                    |
| `TPoint2D`                | `TPoint2D`    | `TPoint2D(lambda s, p: (s.value + p.x, s.value + p.y), s=self, p=other)` |
| `Matrix2D`                | `Matrix2D`    | `Matrix2D(self.value + other.data)`                                      |
| `TMatrix2D`               | `TMatrix2D`   | `TMatrix2D(lambda s, m: m.data + s.value, s=self, m=other)`              |
| Any other type            | ❌ `TypeError` | Operation not supported                                                  |

### Examples

```python
a = Scalar(5)
b = Scalar(3)

print(a + b)  # ➝ Scalar(8)

p = Point2D(1, 2)
print(a + p)  # ➝ Point2D(6, 7)

t = TScalar(lambda s: s.value * 2, s=a)
print(t.get())  # ➝ 10
```

The magic of `TScalar` is that if `a` changes, `t` will automatically update the next time it is evaluated or rendered.
````
