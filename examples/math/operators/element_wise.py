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
import pixelprism.math.functional as E
import pixelprism.math.render as render


# Variables
x = pm.var("x", dtype=pm.DType.FLOAT32, shape=())
y = pm.var("y", dtype=pm.DType.FLOAT32, shape=())

# Math equations
equations = dict()
equations['z1'] = E.add(x, y)
equations['z2'] = E.sub(x, y)
equations['z3'] = E.mul(x, y)
equations['z4'] = E.div(x, y)
equations['z5'] = E.pow(x, y)
equations['z6'] = E.sqrt(x)
equations['z7'] = E.abs(x)
equations['z8'] = E.exp(x)
equations['z9'] = E.exp2(x)
equations['z10'] = E.expm1(x)
equations['z11'] = E.log(x)
equations['z12'] = E.log2(x)
equations['z13'] = E.log10(x)
equations['z14'] = E.cbrt(x)
equations['z15'] = E.square(x)
equations['z16'] = E.reciprocal(x)
equations['z17'] = E.deg2rad(x)
equations['z18'] = E.rad2deg(x)
equations['z19'] = E.abs(x)
equations['z20'] = E.absolute(x)
equations['z21'] = E.neg(x)
equations['z22'] = E.sub(y, equations['z4'])
equations['z23'] = E.mul(equations['z1'], equations['z22'])
equations['z24'] = E.div(equations['z1'], equations['z22'])
equations['z25'] = E.mul(equations['z12'], equations['z9'])

# Show latex
for name, equation in equations.items():
    print(f"Latex of {name}: {render.to_latex(equation)}")
# end for

# Set value and evaluate
with pm.new_context():
    pm.set_value("x", 2.4)
    pm.set_value("y", 2.2)
    for name, equation in equations.items():
        print(f"Equation evaluation {name}: {equation.eval()}")
    # end for
# end with
