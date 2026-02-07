

import pixelprism.math as pm


x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))


print(x)
print(y)


my_eq = (x + 1) + y

print(my_eq)

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
    print(my_eq.eval())
# end with

