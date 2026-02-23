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

# Imports
from __future__ import annotations
from typing import List, Optional, Union, Dict, Mapping, Sequence, TYPE_CHECKING

from .math_base import MathBase
from .math_exceptions import SymbolicMathValidationError, SymbolicMathNotImplementedError
from .math_leaves import Constant, MathLeaf
from .mixins import ExpressionMixin
from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .typing import MathExpr, LeafKind, SimplifyOptions
from .random import rand_name

if TYPE_CHECKING:
    from .math_leaves import Variable
# end if


__all__ = [
    "SliceExpr",
]


class SliceExpr(
    MathBase,
    ExpressionMixin,
    MathExpr
):
    """
    Immutable representation of a Python-style slice using :class:`MathExpr` bounds.

    The object mirrors the behavior of Python's native ``slice`` (and thereby
    NumPy's slicing convention) but stores each bound as an expression so
    transformations can reason about those values symbolically.  Bounds must be
    constant integer expressions so they can be materialized where necessary.

    Parameters
    ----------
    start : MathExpr or int or None, optional
        Inclusive lower bound of the slice. ``None`` matches Python/NumPy semantics.
    stop : MathExpr or int or None, optional
        Exclusive upper bound of the slice. ``None`` matches Python/NumPy semantics.
    step : MathExpr or int or None, optional
        Slice stride. ``None`` matches Python/NumPy semantics.

    Attributes
    ----------
    start : Optional[MathExpr]
        Symbolic representation of the ``start`` bound.
    stop : Optional[MathExpr]
        Symbolic representation of the ``stop`` bound.
    step : Optional[MathExpr]
        Symbolic representation of the ``step`` bound.

    Notes
    -----
    The evaluation rules match ``slice`` and ``numpy.s_`` exactly: ``start`` is
    inclusive, ``stop`` is exclusive, and ``step`` cannot evaluate to zero.  All
    bounds must evaluate to scalar integer tensors, so their Python equivalents
    can be emitted where NumPy-compatible indexing is required.
    """

    __slots__ = ("_start", "_stop", "_step")

    def __init__(
            self,
            start: Optional[Union[MathExpr, int]] = None,
            stop: Optional[Union[MathExpr, int]] = None,
            step: Optional[Union[MathExpr, int]] = None
    ):
        """
        Initialize a new :class:`SliceExpr`.

        Parameters
        ----------
        start : MathExpr or int or None, optional
            Inclusive lower bound of the slice.
        stop : MathExpr or int or None, optional
            Exclusive upper bound of the slice.
        step : MathExpr or int or None, optional
            Slice stride.
        """
        super(SliceExpr, self).__init__(name=rand_name("slice"))
        self._start: Optional[MathExpr] = self._coerce_bound("start", start)
        self._stop: Optional[MathExpr] = self._coerce_bound("stop", stop)
        self._step: Optional[MathExpr] = self._coerce_bound("step", step)

        if self._step and self._step.eval() == 0:
            raise SymbolicMathValidationError(f"Slice step cannot be zero.")
        # end if
    # end def __init__

    # region PROPERTIES

    @property
    def start(self) -> Optional[MathExpr]:
        """
        Returns
        -------
        Optional[MathExpr]
            Symbolic expression for the ``start`` bound.
        """
        return self._start
    # end def start

    @property
    def stop(self) -> Optional[MathExpr]:
        """
        Returns
        -------
        Optional[MathExpr]
            Symbolic expression for the ``stop`` bound.
        """
        return self._stop
    # end def stop

    @property
    def step(self) -> Optional[MathExpr]:
        """
        Returns
        -------
        Optional[MathExpr]
            Symbolic expression for the ``step`` bound.
        """
        return self._step
    # end def step

    @property
    def start_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``start`` when defined.
        """
        return self._expr_to_python(self._start)
    # end def start_value

    @property
    def stop_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``stop`` when defined.
        """
        return self._expr_to_python(self._stop)
    # end def stop_value

    @property
    def step_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``step`` when defined.
        """
        return self._expr_to_python(self._step)
    # end def step_value

    @property
    def as_slice(self) -> slice:
        """
        Returns
        -------
        slice
            Native ``slice`` object mirroring ``self``.
        """
        start_val = self.start_value if self.start is not None else None
        stop_val = self.stop_value if self.stop is not None else None
        step_val = self.step_value if self.step is not None else None

        if start_val is not None and type(start_val) != int:
            raise SymbolicMathValidationError(f"Invalid start value: {start_val}")
        # end if

        if stop_val is not None and type(stop_val) != int:
            raise SymbolicMathValidationError(f"Invalid stop value: {stop_val}")
        # end if

        if step_val is not None and type(step_val) != int:
            raise SymbolicMathValidationError(f"Invalid step value: {step_val}")
        # end if

        return slice(start_val, stop_val, step_val)
    # end def as_slice

    # endregion PROPERTIES

    # region MATH_EXPR

    @property
    def shape(self) -> Shape:
        return Shape(dims=())
    # end def shape

    @property
    def dtype(self) -> DType:
        return DType.Z
    # end def dtype

    @property
    def name(self) -> str:
        return "slice"
    # end def name

    @property
    def rank(self) -> int:
        return 0
    # end def rank

    def eval(self) -> Tensor:
        raise SymbolicMathNotImplementedError("SliceExpr.eval() is not implemented.")
    # end def eval

    def diff(self, wrt: "Variable") -> MathExpr:
        raise SymbolicMathNotImplementedError("SliceExpr does not support differentiation.")
    # end def diff

    def variables(self) -> List["Variable"]:
        out = []
        for child in (self._start, self._stop, self._step):
            if child is not None:
                out.extend(child.variables())
            # end if
        # end for
        unique = {}
        for var in out:
            unique[id(var)] = var
        # end for
        return list(unique.values())
    # end def variables

    def constants(self) -> List["Constant"]:
        out = []
        for child in (self._start, self._stop, self._step):
            if child is not None:
                out.extend(child.constants())
            # end if
        # end for
        unique = {}
        for const in out:
            unique[id(const)] = const
        # end for
        return list(unique.values())
    # end def constants

    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        return any(
            child.contains(leaf, by_ref=by_ref, check_operator=check_operator, look_for=look_for)
            for child in (self._start, self._stop, self._step)
            if child is not None
        )
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, "Variable"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(
            variable,
            by_ref=by_ref,
            check_operator=check_operator,
            look_for=LeafKind.VARIABLE,
        )
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, "Constant"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(
            constant,
            by_ref=by_ref,
            check_operator=check_operator,
            look_for=LeafKind.CONSTANT,
        )
    # end def contains_constant

    def simplify(self, options: SimplifyOptions | None = None) -> MathExpr:
        start = self._start.simplify(options) if self._start is not None else None
        stop = self._stop.simplify(options) if self._stop is not None else None
        step = self._step.simplify(options) if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def simplify

    def canonicalize(self) -> MathExpr:
        start = self._start.canonicalize() if self._start is not None else None
        stop = self._stop.canonicalize() if self._stop is not None else None
        step = self._step.canonicalize() if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def canonicalize

    def fold_constants(self) -> MathExpr:
        start = self._start.fold_constants() if self._start is not None else None
        stop = self._stop.fold_constants() if self._stop is not None else None
        step = self._step.fold_constants() if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def fold_constants

    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True,
    ) -> MathExpr:
        for old_expr, new_expr in mapping.items():
            if by_ref and old_expr is self:
                return new_expr
            # end if
            if (not by_ref) and old_expr == self:
                return new_expr
            # end if
        # end for
        start = self._start.substitute(mapping, by_ref=by_ref) if self._start is not None else None
        stop = self._stop.substitute(mapping, by_ref=by_ref) if self._stop is not None else None
        step = self._step.substitute(mapping, by_ref=by_ref) if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        start = self._start.renamed(old_name, new_name) if self._start is not None else None
        stop = self._stop.renamed(old_name, new_name) if self._stop is not None else None
        step = self._step.renamed(old_name, new_name) if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def renamed

    def eq_tree(self, other: MathExpr) -> bool:
        if not isinstance(other, SliceExpr):
            return False
        # end if
        return (
            SliceExpr._same_child(self._start, other._start)
            and SliceExpr._same_child(self._stop, other._stop)
            and SliceExpr._same_child(self._step, other._step)
        )
    # end def eq_tree

    def equivalent(self, other: MathExpr) -> bool:
        return self.eq_tree(other)
    # end def equivalent

    def is_constant(self) -> bool:
        return not self.is_variable()
    # end def is_constant

    def is_variable(self) -> bool:
        return any(
            child.is_variable()
            for child in (self._start, self._stop, self._step)
            if child is not None
        )
    # end def is_variable

    def is_node(self) -> bool:
        return False
    # end def is_node

    def is_leaf(self) -> bool:
        return False
    # end def is_leaf

    def depth(self) -> int:
        child_depths = [
            child.depth()
            for child in (self._start, self._stop, self._step)
            if child is not None
        ]
        return (max(child_depths) + 1) if child_depths else 1
    # end def depth

    def copy(self, deep: bool = False) -> MathExpr:
        if not deep:
            return SliceExpr(start=self._start, stop=self._stop, step=self._step)
        # end if
        start = self._start.copy(deep=True) if self._start is not None else None
        stop = self._stop.copy(deep=True) if self._stop is not None else None
        step = self._step.copy(deep=True) if self._step is not None else None
        return SliceExpr(start=start, stop=stop, step=step)
    # end def copy

    def __str__(self):
        """
        Returns
        -------
        str
            Human-readable description of the slice expression.
        """
        s = f"{self.start_value}:{self.stop_value}:{self.step_value}"
        s = s.replace("None", "")
        return s
    # end def __str__

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Human-readable representation showing resolved Python bounds.
        """
        return f"SliceExpr(start={self.start_value}, stop={self.stop_value}, step={self.step_value})"
    # end def __repr__

    @staticmethod
    def _same_child(left: Optional[MathExpr], right: Optional[MathExpr]) -> bool:
        if left is None or right is None:
            return left is right
        # end if
        if hasattr(left, "eq_tree"):
            return left.eq_tree(right)
        # end if
        return left == right
    # end def _same_child

    # endregion MATH_EXPR

    # region PUBLIC

    def to_slice(self) -> slice:
        """
        Returns
        -------
        slice
            Native ``slice`` object mirroring ``self``.
        """
        return self.as_slice
    # end def to_slice

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def create(
            start: Optional[Union[MathExpr, int]] = None,
            stop: Optional[Union[MathExpr, int]] = None,
            step: Optional[Union[MathExpr, int]] = None
    ) -> "SliceExpr":
        """
        Instantiate a :class:`SliceExpr` from Python or symbolic bounds.

        Parameters
        ----------
        start : MathExpr or int or None, optional
            Inclusive lower bound of the slice.
        stop : MathExpr or int or None, optional
            Exclusive upper bound of the slice.
        step : MathExpr or int or None, optional
            Stride applied between slice elements.

        Returns
        -------
        SliceExpr
            Newly constructed slice expression.
        """
        return SliceExpr(start=start, stop=stop, step=step)
    # end def create

    @staticmethod
    def from_slice(py_slice: slice) -> "SliceExpr":
        """
        Instantiate a :class:`SliceExpr` from a Python ``slice``.

        Parameters
        ----------
        py_slice : slice
            Python slice whose bounds should be mirrored.

        Returns
        -------
        SliceExpr
            Symbolic slice equivalent to ``py_slice``.
        """
        return SliceExpr(
            start=py_slice.start,
            stop=py_slice.stop,
            step=py_slice.step
        )
    # end def from_slice

    @staticmethod
    def _coerce_bound(
            name: str,
            value: Optional[Union[MathExpr, int]]
    ) -> Optional[MathExpr]:
        """
        Normalize Python and symbolic bounds into :class:`MathExpr` instances.

        Parameters
        ----------
        name : str
            Bound name for diagnostics.
        value : Optional[Union[MathExpr, int]]
            User-provided bound.

        Returns
        -------
        Optional[MathExpr]
            Normalized bound expression, or ``None``.

        Raises
        ------
        TypeError
            If ``value`` is not ``None``, ``int``, or :class:`MathExpr`.
        MathExprValidationError
            If ``value`` violates constant scalar integer requirements.
        """
        if value is None:
            return None
        # end if
        if isinstance(value, int):
            tensor = Tensor(data=value, dtype=DType.Z)
            value = Constant(
                name=rand_name(f"{name}_slice_"),
                data=tensor
            )
        # end if
        if not isinstance(value, MathLeaf):
            raise TypeError(f"{name} must be a MathExpr or int, got {type(value)}")
        # end if
        SliceExpr._validate_bound(name, value)
        return value
    # end def _coerce_bound

    @staticmethod
    def _validate_bound(name: str, expr: MathExpr) -> None:
        """
        Validate that ``expr`` is a scalar integer constant expression.

        Parameters
        ----------
        name : str
            Bound identifier for error messages.
        expr : MathExpr
            Expression to validate.

        Raises
        ------
        MathExprValidationError
            If ``expr`` is not constant, not scalar, or not integer typed.
        """
        if not expr.is_constant():
            raise SymbolicMathValidationError(f"{name} must be composed of constants.")
        # end if
        if expr.dtype not in {DType.Z}:
            raise SymbolicMathValidationError(f"{name} must be an integer expression, got {expr.dtype}")
        # end if
        if expr.shape.dims != ():
            raise SymbolicMathValidationError(f"{name} must be scalar, got shape {expr.shape}")
        # end if
    # end def _validate_bound

    @staticmethod
    def _expr_to_python(expr: Optional[MathExpr]) -> Optional[int]:
        """
        Convert a symbolic bound to its Python integer value.

        Parameters
        ----------
        expr : Optional[MathExpr]
            Symbolic expression representing a slice bound.

        Returns
        -------
        Optional[int]
            Python integer bound or ``None`` when ``expr`` is ``None``.
        """
        if expr is None:
            return None
        # end if
        tensor = expr.eval()
        return int(tensor.value.item())
    # end def _expr_to_python

    # endregion STATIC

# end class SliceExpr
