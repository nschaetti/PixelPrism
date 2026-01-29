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
"""
Demonstrate the tensor builder operator in a plain Python script.
"""

from __future__ import annotations

import pixelprism.math as pm
import pixelprism.math.functional as F


def _describe(label: str, expr) -> None:
    """Evaluate ``expr`` and print its metadata."""
    tensor = expr.eval()
    print(f"{label}: shape={tensor.shape.dims}, dtype={tensor.dtype}")
    print(tensor.value)
    print()
# end def _describe


def main() -> None:
    """Create a few tensors composed of constants, variables, and expressions."""
    with pm.new_context():
        scalar_alpha = pm.const("builder_scalar_alpha", data=1.5, dtype=pm.DType.FLOAT32)
        scalar_beta = pm.const("builder_scalar_beta", data=-2.0, dtype=pm.DType.FLOAT64)
        scalar_gamma = pm.const("builder_scalar_gamma", data=3, dtype=pm.DType.INT32)
        builder_var = pm.var("builder_gain", dtype=pm.DType.FLOAT32, shape=())

        pm.set_value(builder_var.name, 0.5)
        basic_vector = F.build_tensor([scalar_alpha, scalar_beta, 7])
        _describe("basic_vector", basic_vector)

        pm.set_value(builder_var.name, 1.75)
        expression_vector = F.build_tensor(
            [
                builder_var,
                scalar_alpha + scalar_beta,
                F.mul(builder_var, scalar_gamma),
                scalar_gamma + 0.5,
            ]
        )
        _describe("expression_vector", expression_vector)

        pm.set_value(builder_var.name, -0.25)
        matrix_example = F.build_tensor(
            [
                scalar_alpha,
                builder_var,
                scalar_beta,
                builder_var + scalar_gamma,
            ],
            shape=(2, 2),
        )
        _describe("matrix_example", matrix_example)

        tensor3d_example = F.build_tensor(
            [
                scalar_alpha,
                builder_var,
                scalar_beta,
                scalar_gamma,
                builder_var + scalar_gamma,
                scalar_alpha - scalar_beta,
                0.0,
                builder_var * -1.0,
            ],
            shape=(2, 2, 2),
        )
        _describe("tensor3d_example", tensor3d_example)
# end def main


if __name__ == "__main__":
    main()
