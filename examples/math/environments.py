
import pixelprism.math as pm
# from pixelprism.math import root, context, new_context, set_value, create_variable, remove_variable

print(pm.root())
print(pm.context())

print(pm.context().items())

pm.set_value("x", pm.Tensor.from_list([1, 2, 3], dtype=pm.DType.INT32))
pm.set_value("y", pm.Tensor.from_list([4, 5, 6], dtype=pm.DType.INT32))
pm.set_value("phi", pm.Tensor.from_list([7, 8, 9], dtype=pm.DType.INT32))

for k, v in pm.root().items():
    print(f"{k}: {v}")
# end for

with pm.new_context():
    print(pm.context())
    pm.set_value("x", pm.Tensor.from_list([10, 20, 30], dtype=pm.DType.INT32))
    pm.create_variable("z")
    pm.remove_variable("phi")
    print(pm.context().items())
# end with

print(pm.context())
print(pm.context().items())
