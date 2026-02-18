

import pixelprism.math as pm
import pixelprism.math.functional as F


x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))


print(x)
print(y)


my_eq = F.pow((x + 1) - y, 2)

print(my_eq)
print(my_eq.dtype)
print(my_eq.shape)

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
    print(my_eq.eval())
# end with

