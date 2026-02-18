
import pixelprism.math as pm
import pixelprism.math.functional as F

# Constant shapes
n = pm.const("N", dtype=pm.DType.Z, data=2)
m = pm.const("M", dtype=pm.DType.Z, data=3)

#
# Matrix multiplication
#

x = pm.var("x", dtype=pm.DType.R, shape=(2, n))
y = pm.var("y", dtype=pm.DType.R, shape=(n, m))

matmul_expr = F.matmul(x, y)

print("Matrix multiplication")
print(f"x: {x}")
print(f"y: {y}")
print(f"matmul_expr: {matmul_expr}")
print(f"matmul_expr shape: {matmul_expr.shape}")

with pm.new_context():
    pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([[5, 6, 4], [7, 8, 5]], dtype=pm.DType.R))
    print(f"matmul_expr: {matmul_expr.eval()}")
# end with
print("")

#
# Dot product
#

x = pm.var("x", dtype=pm.DType.R, shape=(m,))
y = pm.var("y", dtype=pm.DType.R, shape=(m,))

dot_expr = F.dot(x, y)

print("Dot product")
print(f"x: {x}")
print(f"y: {y}")
print(f"dot_expr: {dot_expr}")
print(f"dot_expr shape: {dot_expr.shape}")

with pm.new_context():
    pm.set_value("x", pm.tensor([1, 2, 3], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([4, 5, 6], dtype=pm.DType.R))
    print(f"dot_expr: {dot_expr.eval()}")
# end with
print("")

#
# Outer product
#

outer_expr = F.outer(x, y)

print("Outer product")
print(f"x: {x}")
print(f"y: {y}")
print(f"outer_expr: {outer_expr}")
print(f"outer_expr shape: {outer_expr.shape}")

with pm.new_context():
    pm.set_value("x", pm.tensor([1, 2, 3], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([4, 5, 6], dtype=pm.DType.R))
    print(f"outer_expr: {outer_expr.eval()}")
# end with
print("")

#
# Trace
#

trace_expr = F.trace(outer_expr)

print("Trace")
print(f"outer_expr: {outer_expr}")
print(f"trace_expr: {trace_expr}")
print(f"trace_expr shape: {trace_expr.shape}")

with pm.new_context():
    pm.set_value("x", pm.tensor([1, 2, 3], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([4, 5, 6], dtype=pm.DType.R))
    pm.set_value("outer_expr", outer_expr.eval())
    print(f"trace_expr: {trace_expr.eval()}")
# end with
print("")

#
# Transpose
#

transpose_expr = F.transpose(outer_expr)

print("Transpose")
print(f"outer_expr: {outer_expr}")
print(f"transpose_expr: {transpose_expr}")
print(f"transpose_expr shape: {transpose_expr.shape}")

with pm.new_context():
    pm.set_value("x", pm.tensor([1, 2, 3], dtype=pm.DType.R))
    pm.set_value("y", pm.tensor([4, 5, 6], dtype=pm.DType.R))
    pm.set_value("outer_expr", outer_expr.eval())
    print(f"outer_expr: {outer_expr.eval()}")
    print(f"transpose_expr: {transpose_expr.eval()}")
# end with
print("")

#
# Determinant
#

x = pm.var("x", dtype=pm.DType.R, shape=(2, n))
det_expr = F.det(x)

print("Determinant")
print(f"x: {x}")
print(f"det_expr: {det_expr}")

with pm.new_context():
    pm.set_value("x", pm.tensor([[2, 0], [0, 2]], dtype=pm.DType.R))
    print(f"x: {x.eval()}")
    print(f"det_expr: {det_expr.eval()}")
# end with
print("")

#
# Inverse
#

inverse_expr = F.inverse(x)

print("Inverse")
print(f"x: {x}")
print(f"inverse_expr: {inverse_expr}")

with pm.new_context():
    pm.set_value("x", pm.tensor([[2, 0], [0, 2]], dtype=pm.DType.R))
    print(f"x: {x.eval()}")
    print(f"inverse_expr: {inverse_expr.eval()}")
# end with
print("")

#
# Norm
#

x = pm.var("x", dtype=pm.DType.R, shape=(m,))
norm_expr = F.norm(x)

print("Norm")
print(f"x: {x}")
print(f"norm_expr: {norm_expr}")

with pm.new_context():
    pm.set_value("x", pm.tensor([2, 0, 0], dtype=pm.DType.R))
    print(f"x: {x.eval()}")
    print(f"norm_expr: {norm_expr.eval()}")
# end with

print("")

#
# Infty norm
#

norm_inf_expr = F.infty_norm(x)

print(f"x: {x}")
print(f"norm_inf_expr: {norm_inf_expr}")

with pm.new_context():
    pm.set_value("x", pm.tensor([2, 0, 0], dtype=pm.DType.R))
    print(f"x: {x.eval()}")
    print(f"norm_inf_expr: {norm_inf_expr.eval()}")
# end with
print("")

#
# Frobenius norm
#

x = pm.var("x", dtype=pm.DType.R, shape=(n,n))
norm_fro_expr = F.frobenius_norm(x)
print(f"x: {x}")
print(f"norm_fro_expr: {norm_fro_expr}")

with pm.new_context():
    pm.set_value("x", pm.tensor([[2, 0], [0, 1]], dtype=pm.DType.R))
    print(f"x: {x.eval()}")
    print(f"norm_fro_expr: {norm_fro_expr.eval()}")
# end with
