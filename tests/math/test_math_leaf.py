

import pytest
from pixelprism.math.dtype import DType
from pixelprism.math.math_expr import MathLeaf
from pixelprism.math.shape import Shape


class DemoLeaf(MathLeaf):
    """Concrete MathLeaf used to validate base-class behavior."""

    def __init__(self, name: str, data: float, mutable: bool = True):
        super().__init__(
            name=name,
            data=data,
            dtype=DType.FLOAT64,
            shape=Shape(()),
            mutable=mutable,
        )
        self.set_calls = []

    def _set(self, data):
        self.set_calls.append(data)
        self._data = data


def test_math_leaf_eval_prefers_stored_data_when_not_overridden():
    leaf = DemoLeaf(name="bias", data=1.5)
    assert DemoLeaf.arity == 0
    assert leaf.is_leaf()
    assert not leaf.is_node()
    assert leaf.eval() == pytest.approx(1.5)


def test_math_leaf_eval_can_be_overridden_by_context():
    leaf = DemoLeaf(name="weight", data=2.0)
    assert leaf.eval(weight=42.0) == 42.0
    # kwargs that do not match the name fall back to the stored value
    assert leaf.eval(unused=13.0) == pytest.approx(2.0)


def test_math_leaf_set_updates_value_for_mutable_nodes():
    leaf = DemoLeaf(name="trainable", data=0.0)
    leaf.set(7.0)
    assert leaf.eval() == pytest.approx(7.0)
    assert leaf.set_calls == [7.0]


def test_math_leaf_set_raises_for_immutable_nodes():
    leaf = DemoLeaf(name="frozen", data=3.0, mutable=False)
    with pytest.raises(RuntimeError):
        leaf.set(9.0)
    assert leaf.eval() == pytest.approx(3.0)
    assert leaf.set_calls == []
