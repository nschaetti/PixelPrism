
import pixelprism.math as pm
import pixelprism.math.functional as F

# Constant shapes
n = pm.const("N", dtype=pm.DType.Z, data=2)
m = pm.const("M", dtype=pm.DType.Z, data=3)

x = pm.var("x", dtype=pm.DType.R, shape=(3, n))
y = pm.var("y", dtype=pm.DType.R, shape=(m, n))
z = pm.var("z", dtype=pm.DType.R, shape=(m, n))


#
# Addition and substraction
#

add_sub_expr = x + y - z

print(f"Add and sub expr: {add_sub_expr}")
print(f"Add and sub expr shape: {add_sub_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    pm.set_value("z", [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
    print(f"Add and sub expr: {add_sub_expr.eval()}")
# end with
print("")


#
# Multiplication and division
#

mult_div_expr = x * y / z

print(f"Mult and div expr: {mult_div_expr}")
print(f"Mult and div expr shape: {mult_div_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    pm.set_value("z", [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
    print(f"Mult and div expr: {mult_div_expr.eval()}")
# end with
print("")


#
# Power
#

power_expr = x ** y

print(f"Power expr: {power_expr}")
print(f"Power expr shape: {power_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    print(f"Power expr: {power_expr.eval()}")
# end with


#
# Negation
#

neg_expr = -x

print(f"Neg expr: {neg_expr}")
print(f"Neg expr shape: {neg_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Neg expr: {neg_expr.eval()}")
# end with
print("")


#
# Exponential
#

exp_expr = F.exp(x)

print(f"Exp expr: {exp_expr}")
print(f"Exp expr shape: {exp_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Exp expr: {exp_expr.eval()}")
# end with
print("")


#
# Exponential base 2
#

exp2_expr = F.exp2(x)

print(f"Exp2 expr: {exp2_expr}")
print(f"Exp2 expr shape: {exp2_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Exp2 expr: {exp2_expr.eval()}")
# end with
print("")


#
# Exponential minus 1
#

expm1_expr = F.expm1(x)

print(f"Expm1 expr: {expm1_expr}")
print(f"Expm1 expr shape: {expm1_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Expm1 expr: {expm1_expr.eval()}")
# end with
print("")


#
# Logarithm (natural)
#

log_expr = F.log(x)

print(f"Log expr: {log_expr}")
print(f"Log expr shape: {log_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Log expr: {log_expr.eval()}")
# end with
print("")


#
# Logarithm (1 + x)
#

log1p_expr = F.log1p(x)

print(f"Log1p expr: {log1p_expr}")
print(f"Log1p expr shape: {log1p_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Log1p expr: {log1p_expr.eval()}")
# end with
print("")


#
# Logarithm base 2
#

log2_expr = F.log2(x)

print(f"Log2 expr: {log2_expr}")
print(f"Log2 expr shape: {log2_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Log2 expr: {log2_expr.eval()}")
# end with
print("")


#
# Logarithm base 10
#

log10_expr = F.log10(x)

print(f"Log10 expr: {log10_expr}")
print(f"Log10 expr shape: {log10_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Log10 expr: {log10_expr.eval()}")
# end with
print("")


#
# Square root
#

sqrt_expr = F.sqrt(x)

print(f"Sqrt expr: {sqrt_expr}")
print(f"Sqrt expr shape: {sqrt_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Sqrt expr: {sqrt_expr.eval()}")
# end with
print("")


#
# Square
#

square_expr = F.square(x)

print(f"Square expr: {square_expr}")
print(f"Square expr shape: {square_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Square expr: {square_expr.eval()}")
# end with
print("")


#
# Cubic root
#

cbrt_expr = F.cbrt(x)

print(f"Cbrt expr: {cbrt_expr}")
print(f"Cbrt expr shape: {cbrt_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Cbrt expr: {cbrt_expr.eval()}")
# end with
print("")


#
# Reciprocal
#

reciprocal_expr = F.reciprocal(x)

print(f"Reciprocal expr: {reciprocal_expr}")
print(f"Reciprocal expr shape: {reciprocal_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Reciprocal expr: {reciprocal_expr.eval()}")
# end with
print("")


#
# Degrees to radians
#

deg2rad_expr = F.deg2rad(x)

print(f"Deg2rad expr: {deg2rad_expr}")
print(f"Deg2rad expr shape: {deg2rad_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[0.0, 30.0], [45.0, 60.0], [90.0, 180.0]])
    print(f"Deg2rad expr: {deg2rad_expr.eval()}")
# end with
print("")


#
# Radians to degrees
#

rad2deg_expr = F.rad2deg(x)

print(f"Rad2deg expr: {rad2deg_expr}")
print(f"Rad2deg expr shape: {rad2deg_expr.shape}")

with pm.new_context():
    pm.set_value("x", [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(f"Rad2deg expr: {rad2deg_expr.eval()}")
# end with
print("")


#
# Absolute
#

absolute_expr = F.absolute(y - z)

print(f"Absolute expr: {absolute_expr}")
print(f"Absolute expr shape: {absolute_expr.shape}")

with pm.new_context():
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    pm.set_value("z", [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
    print(f"Absolute expr: {absolute_expr.eval()}")
# end with
print("")


#
# Abs alias
#

abs_expr = F.abs(y - z)

print(f"Abs expr: {abs_expr}")
print(f"Abs expr shape: {abs_expr.shape}")

with pm.new_context():
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    pm.set_value("z", [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
    print(f"Abs expr: {abs_expr.eval()}")
# end with
print("")
