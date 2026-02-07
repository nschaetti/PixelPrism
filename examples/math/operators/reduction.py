# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2025 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import pixelprism.math as pm
import pixelprism.math.functional as R
import pixelprism.math.render as render


# Variables
x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))
n = pm.var("n", dtype=pm.DType.Z, shape=())
i = pm.var("i", dtype=pm.DType.Z, shape=())

# Math equations
z1 = R.sum(x, axis=0)
z2 = R.sum(y, axis=1)
z3 = z1 + z2
z4 = R.summation(i * n, 1, 10, "i")
prod_idx = pm.var("prod_idx", dtype=pm.DType.Z, shape=())
z5 = R.product(prod_idx + 1, lower=1, upper=4, i="prod_idx")

# Show latex
print(f"Latex of z1: {render.to_latex(z1)}")
print(f"Latex of z2: {render.to_latex(z2)}")
print(f"Latex of z3: {render.to_latex(z3)}")
print(f"Latex of z4: {render.to_latex(z4)}")
print(f"Latex of z5: {render.to_latex(z5)}")

# Set value and evaluate
with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0]])
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0]])
    pm.set_value("n", 2)
    pm.set_value("prod_idx", 0)
    print(f"Equation evaluation z1: {z1.eval()}")
    print(f"Equation evaluation z2: {z2.eval()}")
    print(f"Equation evaluation z3: {z3.eval()}")
    print(f"Equation evaluation z4: {z4.eval()}")
    print(f"Equation evaluation z5: {z5.eval()}")
# end with

outer_idx = pm.var("outer_idx", dtype=pm.DType.Z, shape=())
inner_idx = pm.var("inner_idx", dtype=pm.DType.Z, shape=())
weight = pm.var("nested_weight", dtype=pm.DType.R, shape=())
bias = pm.const("nested_bias", data=1.0, dtype=pm.DType.R)

inner_body = (inner_idx + outer_idx) * weight + bias
inner_sum = R.summation(
    op1=inner_body,
    lower=outer_idx,
    upper=outer_idx + pm.const("inner_span", data=3, dtype=pm.DType.Z),
    i="inner_idx"
)
nested_expr = R.summation(
    op1=inner_sum * (outer_idx + pm.const("outer_offset", data=1, dtype=pm.DType.Z)),
    lower=1,
    upper=3,
    i="outer_idx"
)

print(render.to_latex(nested_expr))

with pm.new_context():
    pm.set_value("nested_weight", 0.5)
    expected = 0.0
    for outer in range(1, 4):
        inner_total = 0.0
        for inner in range(outer, outer + 2):
            inner_total += (inner + outer) * 0.5 + 1.0
        # end for
        expected += inner_total * (outer + 1)
    # end for
    print(nested_expr.eval())
# end with
