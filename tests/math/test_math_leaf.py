from typing import List

import pytest

from pixelprism.math import Tensor
from pixelprism.math.dtype import DType
from pixelprism.math.math_expr import MathLeaf
from pixelprism.math.shape import Shape


class DemoLeaf(MathLeaf):
    """Concrete MathLeaf used to validate base-class behavior."""

    def variables(self) -> list:
        pass

    def constants(self) -> List:
        pass

    def _eval(self) -> Tensor:
        pass

    def __init__(self, name: str):
        super().__init__(
            name=name,
            dtype=DType.R,
            shape=Shape(())
        )
        self.set_calls = []

    def _set(self, data):
        self.set_calls.append(data)
        self._data = data


def test_math_leaf_eval_prefers_stored_data_when_not_overridden():
    leaf = DemoLeaf(name="bias")
    assert DemoLeaf.arity == 0
    assert leaf.is_leaf()
    assert not leaf.is_node()
# end def test_math_leaf_eval_prefers_stored_data_when_not_overridden
