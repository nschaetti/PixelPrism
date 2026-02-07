

import pixelprism.math as pm
import pixelprism.math.functional as F
import pixelprism.math.render as render


x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))
n = pm.var("n", dtype=pm.DType.Z, shape=())

my_eq = F.pow(x + (1 + y) - 12.0, n)

print(render.to_latex(my_eq))

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
    pm.set_value("n", pm.tensor(2, dtype=pm.DType.Z))
    print(my_eq.eval())
# end with

# print(my_eq.eval())

my_eq = F.cbrt(F.log10(x) + (1 + y) - 12.0)

print(render.to_latex(my_eq))

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
    pm.set_value("n", pm.tensor(2, dtype=pm.DType.Z))
    print(my_eq.eval())
# end with

my_eq = F.floor(F.sin(x) + F.cos(y))
# my_eq = F.sin(x) + F.cos(y)

# print(my_eq.children)
print(render.to_latex(my_eq))

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
    pm.set_value("n", pm.tensor(2, dtype=pm.DType.Z))
    print(my_eq.eval())
# end with


