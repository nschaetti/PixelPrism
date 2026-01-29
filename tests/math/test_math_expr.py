import pytest

from pixelprism.math.dtype import DType
from pixelprism.math.math_expr import (
    MathNode,
    MathLeaf,
    MathExprNotImplementedError,
    MathExprOperatorError,
)
from pixelprism.math.shape import Shape
from pixelprism.math.tensor import Tensor
from pixelprism.math.operators import Operator


class DummyLeaf(MathLeaf):
    """Concrete MathLeaf carrying a scalar tensor."""

    def __init__(self, name: str, value: float = 0.0):
        self._data = float(value)
        super().__init__(name=name, dtype=DType.FLOAT32, shape=Shape((1,)))
    # end def __init__

    def _eval(self) -> Tensor:
        return Tensor(data=[self._data], dtype=DType.FLOAT32)
    # end def _eval

    def _set(self, data: float) -> None:
        self._data = float(data)
    # end def _set

    def variables(self) -> list:
        return []
    # end def variables

    def constants(self) -> list:
        return [self]
    # end def constants

# end class DummyLeak


class TrackableMathNode(MathNode):
    """MathExpr subclass that supports weak references for parent tracking."""
    pass
# end class TrackableMathExpr


class DummyOp(Operator):
    """Minimal operator implementation for MathExpr tests."""

    def __str__(self) -> str:
        pass
    # end def __str__

    def __repr__(self) -> str:
        pass
    # end def __repr__

    ARITY = 0
    NAME = "noop"

    def __init__(self, name="noop", arity=0, fn=None):
        self.NAME = name
        self.ARITY = arity
        self._fn = fn or (lambda values: values)
        super().__init__()
    # end def __init__

    def contains(self, expr):
        return False
    # end def contains

    def check_operands(self, operands):
        return len(operands) == self.arity
    # end def check_operands

    def _eval(self, operands):
        values = [
            child.eval().value.item()
            for child in operands
        ]
        result = self._fn(values)
        return Tensor(data=[result], dtype=DType.FLOAT32)
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
        return operands[0].input_shape if operands else Shape(())
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
        expr_cls=MathNode
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

    leaf_like = MathNode(
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
        expr_cls=TrackableMathNode,
    )

    first_eval = node.eval()
    assert isinstance(first_eval, Tensor)
    assert first_eval.value.item() == pytest.approx(30.0)

    child1._set(3.0)
    child2._set(4.0)
    updated_eval = node.eval()
    assert updated_eval.value.item() == pytest.approx(7.0)

    assert isinstance(child1.parents, frozenset)
    assert node in child1.parents
    assert node in child2.parents
# end test test_math_expr_eval_and_parent_registration


def test_math_expr_operator_mismatch_raises():
    child1 = DummyLeaf("one", 1.0)
    child2 = DummyLeaf("two", 2.0)
    bad_op = DummyOp(name="bad", arity=1)

    with pytest.raises(MathExprOperatorError):
        MathNode(
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
    assert rep.startswith(f"<{node.__class__.__name__} #")
    assert "identity" in rep
    assert node.dtype.value in rep
    assert f"c:{node.arity}" in rep

    assert hash(node) == hash(node)
    other = _make_node((child,), name="other", op_name="identity", fn=lambda values: values[0])
    assert node == node
    assert node != other
    assert not node.__eq__(object())
    assert node.__ne__(other)
# end test test_math_expr_repr_hash_and_identity_semantics


def test_math_expr_operator_overloads_dispatch(monkeypatch):
    child = DummyLeaf("only", 1.0)
    node = _make_node((child,), fn=lambda values: values[0])
    other = _make_node((child,), name="other", fn=lambda values: values[0])

    class NonExpr:
        def __add__(self, _):
            return NotImplemented

        def __sub__(self, _):
            return NotImplemented

        def __mul__(self, _):
            return NotImplemented

        def __truediv__(self, _):
            return NotImplemented

        def __pow__(self, _, __=None):
            return NotImplemented
    # end class NonExpr

    non_expr = NonExpr()

    monkeypatch.setattr(MathNode, "add", lambda a, b: ("add", a, b))
    monkeypatch.setattr(MathNode, "sub", lambda a, b: ("sub", a, b))
    monkeypatch.setattr(MathNode, "mul", lambda a, b: ("mul", a, b))
    monkeypatch.setattr(MathNode, "div", lambda a, b: ("div", a, b))
    monkeypatch.setattr(MathNode, "pow", lambda a, b: ("pow", a, b))
    monkeypatch.setattr(MathNode, "neg", lambda a: ("neg", a))

    assert node + other == ("add", node, other)
    assert non_expr + node == ("add", non_expr, node)

    assert node - other == ("sub", node, other)
    assert non_expr - node == ("sub", non_expr, node)

    assert node * other == ("mul", node, other)
    assert non_expr * node == ("mul", non_expr, node)

    assert node / other == ("div", node, other)
    assert non_expr / node == ("div", non_expr, node)

    assert node ** other == ("pow", node, other)
    assert non_expr ** node == ("pow", non_expr, node)
    assert ("neg", node) == -node

    with pytest.raises(MathExprNotImplementedError):
        _ = node < other
    with pytest.raises(MathExprNotImplementedError):
        _ = node <= other
    with pytest.raises(MathExprNotImplementedError):
        _ = node > other
    with pytest.raises(MathExprNotImplementedError):
        _ = node >= other
# end test test_math_expr_operator_overloads_dispatch
