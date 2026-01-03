import pytest

from pixelprism.math.dtype import DType
from pixelprism.math.math_expr import MathExpr, MathLeaf
from pixelprism.math.shape import Shape
from pixelprism.math.operators import Operator


class DummyLeaf(MathLeaf):
    """Simple concrete leaf for exercising MathExpr behavior in tests."""

    def __init__(self, name: str, value: float):
        super().__init__(name=name, data=value, dtype=DType.FLOAT32, shape=Shape((1,)))
    # end def __init__

    def _set(self, data):
        self._data = data
    # end def _set

# end class DummyLeak


class TrackableMathExpr(MathExpr):
    """MathExpr subclass that supports weak references for parent tracking."""
    pass
# end class TrackableMathExpr


class DummyOp(Operator):
    """Minimal operator implementation for MathExpr tests."""

    ARITY = 0
    NAME = "noop"

    def __init__(self, name="noop", arity=0, fn=None):
        self.NAME = name
        self.ARITY = arity
        self._fn = fn or (lambda values: values)
        super().__init__()
    # end def __init__

    def _eval(self, values):
        return self._fn(values)
    # end def _eval

    def _backward(self, out_grad, node):
        return tuple(out_grad for _ in node.children)
    # end def _backward

    @classmethod
    def infer_dtype(cls, operands):
        return operands[0].dtype if operands else DType.FLOAT32
    # end def infer_dtype

    @classmethod
    def infer_shape(cls, operands):
        return operands[0].shape if operands else Shape(())
    # end def infer_shape

    @classmethod
    def check_shapes(cls, operands):
        return True
    # end def check_shapes

# end class DummyOp


def _make_node(
        children,
        name="node",
        op_name="sum",
        fn=None,
        expr_cls=MathExpr
):
    return expr_cls(
        name=name,
        op=DummyOp(name=op_name, arity=len(children), fn=fn),
        children=children,
        dtype=DType.FLOAT32,
        shape=Shape((len(children), len(children))),
    )
# end def _make_node


def test_math_expr_basic_properties_and_leaf_detection():
    """Test basic properties of MathExpr and leaf detection.
    """
    child1 = DummyLeaf("left", 1.0)
    child2 = DummyLeaf("right", 2.0)
    node = _make_node(
        (child1, child2),
        name="sum_node",
        fn=lambda values: sum(values),
    )

    assert node.name == "sum_node"
    assert node.op.name == "sum"
    assert node.children == (child1, child2)
    assert node.dtype == DType.FLOAT32
    assert node.shape.dims == (2, 2)
    assert node.arity == 2
    assert node.is_node()
    assert not node.is_leaf()

    leaf_like = MathExpr(
        name="leafish",
        op=None,
        children=(),
        dtype=DType.FLOAT64,
        shape=Shape(()),
    )

    assert leaf_like.is_leaf()
    assert not leaf_like.is_node()
    assert leaf_like.arity == 0
# end test test_math_expr_basic_properties_and_leaf_detection


def test_math_expr_eval_and_parent_registration():
    child1 = DummyLeaf("left", 10.0)
    child2 = DummyLeaf("right", 20.0)

    node = _make_node(
        (child1, child2),
        fn=lambda values: sum(values),
        expr_cls=TrackableMathExpr,
    )
    node._register_as_parent_of(*node.children)

    assert node.eval() == 30.0
    # kwargs override leaf data
    assert node.eval(left=3.0, right=4.0) == 7.0

    assert isinstance(child1.parents, frozenset)
    assert node in child1.parents
    assert node in child2.parents
# end test test_math_expr_eval_and_parent_registration


def test_math_expr_operator_mismatch_raises():
    child1 = DummyLeaf("one", 1.0)
    child2 = DummyLeaf("two", 2.0)
    bad_op = DummyOp(name="bad", arity=1)

    with pytest.raises(TypeError):
        MathExpr(
            name="bad_node",
            op=bad_op,
            children=(child1, child2),
            dtype=DType.FLOAT32,
            shape=Shape((1,)),
        )
    # end with
# end test test_math_expr_operator_mismatch_raises


def test_math_expr_repr_hash_and_identity_semantics(monkeypatch):
    monkeypatch.setattr(Shape, "__str__", lambda self: self.__repr__())
    child = DummyLeaf("only", 5.0)
    node = _make_node(
        (child,),
        name="inspectable",
        op_name="identity",
        fn=lambda values: values[0],
    )

    rep = repr(node)
    assert "MathExpr" in rep
    assert "identity" in rep
    assert f"c:{node.arity}" in rep

    assert hash(node) == hash(node)
    other = _make_node((child,), name="other", op_name="identity", fn=lambda values: values[0])
    assert node == node
    assert node != other
    assert not node.__eq__(object())
    assert node.__ne__(other)
# end test test_math_expr_repr_hash_and_identity_semantics


def test_math_expr_operator_overloads_default_to_none():
    child = DummyLeaf("only", 1.0)
    node = _make_node((child,), fn=lambda values: values[0])
    assert node + 1 is None
    assert 1 + node is None
    assert node - 1 is None
    assert 1 - node is None
    assert node * 2 is None
    assert 2 * node is None
    assert node / 2 is None
    assert 2 / node is None
    assert (node < 0) is None
    assert (node <= 0) is None
    assert (node > 0) is None
    assert (node >= 0) is None
# end test test_math_expr_operator_overloads_default_to_none
