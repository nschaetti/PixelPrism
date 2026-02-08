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
from typing import List, Optional, Union, Dict

from .math_exceptions import SymbolicMathValidationError
from .math_leaves import Constant, MathLeaf
from .mixins import PredicateMixin
from .dtype import DType
from .tensor import Tensor
from .typing import MathExpr
from .random import rand_name


__all__ = [
    "SliceExpr",
]


class SliceExpr(PredicateMixin):
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

    # region PREDICATE_MIXIN

    def is_constant(self):
        """Contains any variables?"""
        return not self.is_variable()
    # end def is_constant

    def is_variable(self):
        """Contains any variables?"""
        has_variable = False
        if self._start is not None and not self._start.is_variable(): has_variable = True
        if self._stop is not None and not self._stop.is_variable(): has_variable = True
        if self._step is not None and not self._step.is_variable(): has_variable = True
        return has_variable
    # end def is_variable

    def variables(self) -> List['Variable']:
        return []
    # end def variables

    def constants(self) -> List['Constant']:
        slice_const = list()
        slice_const.extend(self._start.constants() if self._start else [])
        slice_const.extend(self._stop.constants() if self._stop else [])
        slice_const.extend(self._step.constants() if self._step else [])
        return slice_const
    # end def constantes

    def contains(
            self,
            leaf: Union[str, PredicateMixin],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        children = [o for o in [self._start, self._stop, self._step] if o is not None]
        return any([
            o.contains(leaf, by_ref, check_operator, look_for)
            for o in children
        ])
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(variable, by_ref, check_operator, "var")
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(constant, by_ref, check_operator, "const")
    # end def contains_constant

    def replace(self, old_m: PredicateMixin, new_m: PredicateMixin):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence."""
        if self._start is not None: self._start.replace(old_m, new_m)
        if self._stop is not None: self._stop.replace(old_m, new_m)
        if self._step is not None: self._step.replace(old_m, new_m)
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        if self._start is not None: self._start.rename(old_name, new_name)
        if self._stop is not None: self._stop.rename(old_name, new_name)
        if self._step is not None: self._step.rename(old_name, new_name)
        return {old_name: new_name}
    # end def rename

    # endregion PREDICATE_MIXIN

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

    # region OVERRIDE

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

    # endregion OVERRIDE

# end class SliceExpr

