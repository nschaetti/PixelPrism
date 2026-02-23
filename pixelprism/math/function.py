"""Function wrapper for symbolic math expressions.

This module defines :class:`Function`, a symbolic expression wrapper that keeps
an internal expression body while exposing a function-like call interface.

The wrapper behaves as an atomic symbolic object in expression trees:

- it can be evaluated directly with :meth:`Function.__call__` by providing
  values for its variables,
- it can still be evaluated through :meth:`Function.eval` in the active context,
- and it acts as a simplification barrier (``simplify`` returns itself).
"""

# Imports
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, TYPE_CHECKING, cast

from .math_base import MathBase
from .math_exceptions import SymbolicMathValidationError, SymbolicMathNotImplementedError
from .math_node import MathNode
from .mixins import ExpressionMixin
from .context import new_context, set_value


if TYPE_CHECKING:
    from .typing import MathExpr, SimplifyOptions
    from .tensor import Tensor
    from .math_leaves import Variable, Constant
    from .typing import LeafKind
# end if


__all__ = [
    "Function",
]


class Function(
    MathBase,
    ExpressionMixin,
):
    """Atomic symbolic function built from an expression body.

    A :class:`Function` wraps a symbolic expression and exposes a callable API
    while preserving expression compatibility. The function advertises the same
    dtype/shape as its body and is usable as an operand in larger symbolic
    expressions.

    The wrapper acts as a simplification barrier: calling :meth:`simplify`
    returns ``self`` and does not simplify the internal body.

    Parameters
    ----------
    name : str
        Symbolic name of the function (for display/debug purposes).
    body : MathExpr
        Symbolic expression wrapped by the function.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> x = pm.var("x", dtype=pm.R, shape=())
    >>> y = pm.var("y", dtype=pm.R, shape=())
    >>> f = Function(name="F", body=(x * x) + y)
    >>> f.variable_names
    ('x', 'y')
    >>> out = f(3.0, y=4.0)
    >>> out.value.item()
    13.0
    """

    __slots__ = (
        "_body",
        "_variables_by_name",
        "_variable_names",
    )

    # region CONSTRUCTOR

    def __init__(
            self,
            *,
            name: str,
            body: "MathExpr"
    ) -> None:
        """Initialize a symbolic function from an expression body.

        Parameters
        ----------
        name : str
            Symbolic identifier of the function.
        body : MathExpr
            Expression used as the function body.

        Raises
        ------
        TypeError
            If ``body`` does not expose the minimal MathExpr interface.
        """
        self._validate_body(body)
        super(Function, self).__init__(
            name=name,
            dtype=body.dtype,
            shape=body.shape,
        )
        self._body = body

        unique: "OrderedDict[str, Variable]" = OrderedDict()
        for var in body.variables():
            var_name = var.name
            if var_name is None:
                continue
            # end if
            if var_name not in unique:
                unique[var_name] = var
            # end if
        # end for

        self._variables_by_name = dict(unique)
        self._variable_names = tuple(sorted(self._variables_by_name.keys()))
    # end def __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

    @property
    def body(self) -> "MathExpr":
        """Return the wrapped symbolic expression.

        Returns
        -------
        MathExpr
            Internal expression wrapped by the function.

        Examples
        --------
        >>> import pixelprism.math as pm
        >>> x = pm.var("x", dtype=pm.R, shape=())
        >>> f = Function(name="F", body=x + 1)
        >>> f.body is not None
        True
        """
        return self._body
    # end def body

    @property
    def name(self) -> str:
        """Return the symbolic name of this function.

        Returns
        -------
        str
            Function name.
        """
        return str(self._name)
    # end def name

    @property
    def rank(self) -> int:
        """Return the rank of the function output tensor.

        Returns
        -------
        int
            Rank derived from :attr:`shape`.
        """
        return self.shape.rank
    # end def rank

    @property
    def variables_map(self) -> Mapping[str, "Variable"]:
        """Return variables used by the function body, keyed by name.

        Returns
        -------
        Mapping[str, Variable]
            Mapping from variable name to variable object.

        Notes
        -----
        The mapping preserves one representative variable per name.
        """
        return self._variables_by_name.copy()
    # end def variables_map

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return ordered variable names used for positional binding.

        Returns
        -------
        tuple[str, ...]
            Variable names sorted alphabetically.

        Examples
        --------
        >>> import pixelprism.math as pm
        >>> x = pm.var("x", dtype=pm.R, shape=())
        >>> y = pm.var("y", dtype=pm.R, shape=())
        >>> f = Function(name="F", body=y + x)
        >>> f.variable_names
        ('x', 'y')
        """
        return self._variable_names
    # end def variable_names

    # endregion PROPERTIES

    # region PUBLIC

    def __getattr__(self, item: str) -> "Variable":
        """Resolve variables as dynamic attributes.

        Parameters
        ----------
        item : str
            Attribute name.

        Returns
        -------
        Variable
            Variable matching ``item``.

        Raises
        ------
        AttributeError
            If no variable with that name exists.

        Examples
        --------
        >>> import pixelprism.math as pm
        >>> x = pm.var("x", dtype=pm.R, shape=())
        >>> f = Function(name="F", body=x + 1)
        >>> f.x.name
        'x'
        """
        try:
            return self._variables_by_name[item]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {item!r}.") from exc
        # end try
    # end def __getattr__

    def __call__(self, *args: Any, **kwargs: Any) -> "Tensor":
        """Evaluate the function by binding variables in a fresh context.

        Positional arguments are mapped according to :attr:`variable_names`
        (alphabetical order). Keyword arguments are mapped by variable name.

        Parameters
        ----------
        *args : Any
            Positional values bound in alphabetical variable order.
        **kwargs : Any
            Named values bound by variable name.

        Returns
        -------
        Tensor
            Evaluation result of the wrapped body.

        Raises
        ------
        TypeError
            If argument binding is invalid (missing value, unknown variable,
            duplicate binding, or too many positional arguments).

        Examples
        --------
        >>> import pixelprism.math as pm
        >>> x = pm.var("x", dtype=pm.R, shape=())
        >>> y = pm.var("y", dtype=pm.R, shape=())
        >>> f = Function(name="F", body=(x * x) + y)
        >>> f(3.0, 4.0).value.item()
        13.0
        >>> f(3.0, y=4.0).value.item()
        13.0
        >>> f(x=3.0, y=4.0).value.item()
        13.0
        """
        bindings = self._bind_call_arguments(args=args, kwargs=kwargs)
        with new_context():
            for name in self._variable_names:
                set_value(name, bindings[name])
            # end for
            return self._body.eval()
        # end with
    # end def __call__

    # endregion PUBLIC

    # region MATH_EXPR

    def eval(self) -> "Tensor":
        """Evaluate the function body in the current active context.

        Returns
        -------
        Tensor
            Result of evaluating the wrapped body.

        Examples
        --------
        >>> import pixelprism.math as pm
        >>> x = pm.var("x", dtype=pm.R, shape=())
        >>> f = Function(name="F", body=x + 2)
        >>> with pm.new_context() as scope:
        ...     scope.set("x", 5.0)
        ...     out = f.eval()
        >>> out.value.item()
        7.0
        """
        return self._body.eval()
    # end def eval

    def diff(self, wrt: "Variable") -> "MathExpr":
        """Differentiate the wrapped body with respect to ``wrt``.

        Parameters
        ----------
        wrt : Variable
            Variable used for differentiation.

        Returns
        -------
        MathExpr
            Symbolic derivative expression.
        """
        # return self._body.diff(wrt)
        ff = Function(name=f"d{self.name}/d{wrt.name}", body=self._body.diff(wrt=wrt))
        return cast(MathExpr, cast(object, ff))
    # end def diff

    def variables(self) -> list["Variable"]:
        """Return variables referenced by the function body.

        Returns
        -------
        list[Variable]
            Unique variables by name.
        """
        return [self._variables_by_name[name] for name in self._variable_names]
    # end def variables

    def constants(self) -> list["Constant"]:
        """Return constants referenced by the function body.

        Returns
        -------
        list[Constant]
            Constants found in the wrapped body.
        """
        return list(self._body.constants())
    # end def constants

    def contains(
            self,
            leaf: "str | MathExpr",
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Any = None
    ) -> bool:
        """Test whether an expression or name appears in the function.

        Parameters
        ----------
        leaf : str or MathExpr
            Name or expression to search for.
        by_ref : bool, default False
            If ``True``, require identity match for expression inputs.
        check_operator : bool, default True
            Forwarded to wrapped expression lookups.
        look_for : LeafKind, optional
            Restrict lookup to variables, constants, or both.

        Returns
        -------
        bool
            ``True`` if the target is found.

        Raises
        ------
        SymbolicMathValidationError
            If ``by_ref=True`` and ``leaf`` is provided as a string.
        """
        if by_ref and isinstance(leaf, str):
            raise SymbolicMathValidationError("Cannot perform identity lookup with a string leaf.")
        # end if

        if by_ref and leaf is self:
            return True
        # end if

        if not by_ref and isinstance(leaf, str) and leaf == self.name:
            return True
        # end if

        mode = self._normalize_look_for(look_for)
        if mode == "var":
            return self.contains_variable(variable=cast(Any, leaf), by_ref=by_ref, check_operator=check_operator)
        # end if
        if mode == "const":
            return self.contains_constant(constant=cast(Any, leaf), by_ref=by_ref, check_operator=check_operator)
        # end if
        return self._body.contains(leaf, by_ref=by_ref, check_operator=check_operator, look_for=look_for)
    # end def contains

    def contains_variable(
            self,
            variable: "str | Variable",
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return ``True`` if the function body contains ``variable``.

        Parameters
        ----------
        variable : str or Variable
            Variable name or variable expression.
        by_ref : bool, default False
            If ``True``, require identity match.
        check_operator : bool, default True
            Forwarded to wrapped expression lookups.
        """
        return self._body.contains_variable(
            variable=variable,
            by_ref=by_ref,
            check_operator=check_operator,
        )
    # end def contains_variable

    def contains_constant(
            self,
            constant: "str | Constant",
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return ``True`` if the function body contains ``constant``.

        Parameters
        ----------
        constant : str or Constant
            Constant name or constant expression.
        by_ref : bool, default False
            If ``True``, require identity match.
        check_operator : bool, default True
            Forwarded to wrapped expression lookups.
        """
        return self._body.contains_constant(
            constant=constant,
            by_ref=by_ref,
            check_operator=check_operator,
        )
    # end def contains_constant

    def simplify(self, options: "SimplifyOptions | None" = None) -> "MathExpr":
        """Return ``self`` without simplifying the wrapped body.

        This method intentionally acts as a simplification barrier.

        Parameters
        ----------
        options : SimplifyOptions or None, default None
            Unused for this wrapper.

        Returns
        -------
        MathExpr
            ``self``.
        """
        return self
    # end def simplify

    def canonicalize(self) -> "MathExpr":
        """Return ``self`` unchanged.

        Returns
        -------
        MathExpr
            ``self``.
        """
        return self
    # end def canonicalize

    def fold_constants(self) -> "MathExpr":
        """Return ``self`` unchanged.

        Returns
        -------
        MathExpr
            ``self``.
        """
        return self
    # end def fold_constants

    def substitute(
            self,
            mapping: Mapping["MathExpr", "MathExpr"],
            *,
            by_ref: bool = True
    ) -> "MathExpr":
        """Substitute this function when it matches a mapping key.

        The function body is intentionally not traversed to preserve the
        simplification/substitution barrier semantics.

        Parameters
        ----------
        mapping : Mapping[MathExpr, MathExpr]
            Replacement mapping.
        by_ref : bool, default True
            If ``True``, matching is identity-based; otherwise uses ``==``.

        Returns
        -------
        MathExpr
            Replacement expression if matched, otherwise ``self``.
        """
        for old_expr, new_expr in mapping.items():
            if by_ref and old_expr is self:
                return new_expr
            # end if
            if (not by_ref) and old_expr == self:
                return new_expr
            # end if
        # end for
        return self
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> "MathExpr":
        """Return a new function with a renamed function symbol.

        The wrapped body is intentionally left unchanged.

        Parameters
        ----------
        old_name : str
            Name to replace.
        new_name : str
            Replacement name.

        Returns
        -------
        MathExpr
            New :class:`Function` if ``self.name`` matches ``old_name``,
            otherwise ``self``.
        """
        if self.name == old_name:
            return cast(
                MathExpr,
                cast(
                    object,
                    Function(name=new_name, body=self._body)
                )
            )
        # end if
        return cast(MathExpr, cast(object, self))
    # end def renamed

    def eq_tree(self, other: "MathExpr") -> bool:
        """Return strict tree equality for this wrapper.

        Parameters
        ----------
        other : MathExpr
            Expression to compare with.

        Returns
        -------
        bool
            ``True`` if both expressions are :class:`Function` instances with
            the same name and wrapped body identity.
        """
        if not isinstance(other, Function):
            return False
        # end if
        return (self.name == other.name) and (self._body is other._body)
    # end def eq_tree

    def equivalent(self, other: "MathExpr") -> bool:
        """Return symbolic equivalence for this wrapper.

        Parameters
        ----------
        other : MathExpr
            Expression to compare with.

        Returns
        -------
        bool
            For this wrapper, equivalence is delegated to :meth:`eq_tree`.
        """
        return self.eq_tree(other)
    # end def equivalent

    def is_constant(self) -> bool:
        """Return whether the wrapped body depends only on constants.

        Returns
        -------
        bool
            ``True`` if the wrapped body is constant.
        """
        return self._body.is_constant()
    # end def is_constant

    def is_variable(self) -> bool:
        """Return whether the wrapped body depends on at least one variable.

        Returns
        -------
        bool
            ``True`` if variables are present in the wrapped body.
        """
        return self._body.is_variable()
    # end def is_variable

    def is_node(self) -> bool:
        """Return ``False`` because this wrapper behaves as an atom.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False
    # end def is_node

    def is_leaf(self) -> bool:
        """Return ``True`` because this wrapper behaves as an atom.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True
    # end def is_leaf

    def depth(self) -> int:
        """Return symbolic depth for this atomic wrapper.

        Returns
        -------
        int
            Always ``0``.
        """
        return 0
    # end def depth

    def copy(self, deep: bool = False) -> "MathExpr":
        """Copy the function wrapper.

        Parameters
        ----------
        deep : bool, default False
            If ``True`` and the wrapped body exposes ``copy(deep=...)``, a deep
            body copy is requested.

        Returns
        -------
        MathExpr
            Copied function wrapper.
        """
        if deep and hasattr(self._body, "copy"):
            body_copy = self._body.copy(deep=True)
        else:
            body_copy = self._body
        # end if
        return Function(name=self.name, body=body_copy)
    # end def copy

    def __str__(self) -> str:
        """Return a concise user-facing representation.

        Returns
        -------
        str
            Function symbol (its ``name``).
        """
        return str(self.name)
    # end def __str__

    def __repr__(self) -> str:
        """Return a debug-friendly representation.

        Returns
        -------
        str
            Representation containing class name, identifier, and variables.
        """
        return (
            f"<Function #{self.identifier} {self.name} "
            f"{self.dtype.value} {self.shape} vars:{self._variable_names}>"
        )
    # end def __repr__

    # endregion MATH_EXPR

    # region OVERRIDE

    def __hash__(self) -> int:
        """Return an identity-based hash.

        Returns
        -------
        int
            Hash of the object identifier.
        """
        return hash(self.identifier)
    # end def __hash__

    def __eq__(self, other: object) -> bool:
        """Compare wrapper identity.

        Parameters
        ----------
        other : object
            Object to compare.

        Returns
        -------
        bool
            ``True`` when both wrappers share the same identifier.
        """
        if self is other:
            return True
        # end if
        if not isinstance(other, Function):
            return False
        # end if
        return self.identifier == other.identifier
    # end def __eq__

    def __ne__(self, other: object) -> bool:
        """Return logical negation of :meth:`__eq__`.

        Parameters
        ----------
        other : object
            Object to compare.

        Returns
        -------
        bool
            ``True`` when objects are different by identity.
        """
        return not self.__eq__(other)
    # end def __ne__

    def __add__(self, other: Any) -> MathNode:
        """Create an addition expression with ``other``.

        Parameters
        ----------
        other : Any
            Right operand.

        Returns
        -------
        MathNode
            Symbolic addition node.
        """
        return MathNode.add(cast(Any, self), other)
    # end def __add__

    def __radd__(self, other: Any) -> MathNode:
        """Create a reverse addition expression.

        Parameters
        ----------
        other : Any
            Left operand.

        Returns
        -------
        MathNode
            Symbolic addition node.
        """
        return MathNode.add(other, cast(Any, self))
    # end def __radd__

    def __sub__(self, other: Any) -> MathNode:
        """Create a subtraction expression with ``other``.

        Parameters
        ----------
        other : Any
            Right operand.

        Returns
        -------
        MathNode
            Symbolic subtraction node.
        """
        return MathNode.sub(cast(Any, self), other)
    # end def __sub__

    def __rsub__(self, other: Any) -> MathNode:
        """Create a reverse subtraction expression.

        Parameters
        ----------
        other : Any
            Left operand.

        Returns
        -------
        MathNode
            Symbolic subtraction node.
        """
        return MathNode.sub(other, cast(Any, self))
    # end def __rsub__

    def __mul__(self, other: Any) -> MathNode:
        """Create a multiplication expression with ``other``.

        Parameters
        ----------
        other : Any
            Right operand.

        Returns
        -------
        MathNode
            Symbolic multiplication node.
        """
        return MathNode.mul(cast(Any, self), other)
    # end def __mul__

    def __rmul__(self, other: Any) -> MathNode:
        """Create a reverse multiplication expression.

        Parameters
        ----------
        other : Any
            Left operand.

        Returns
        -------
        MathNode
            Symbolic multiplication node.
        """
        return MathNode.mul(other, cast(Any, self))
    # end def __rmul__

    def __truediv__(self, other: Any) -> MathNode:
        """Create a division expression with ``other``.

        Parameters
        ----------
        other : Any
            Right operand.

        Returns
        -------
        MathNode
            Symbolic division node.
        """
        return MathNode.div(cast(Any, self), other)
    # end def __truediv__

    def __rtruediv__(self, other: Any) -> MathNode:
        """Create a reverse division expression.

        Parameters
        ----------
        other : Any
            Left operand.

        Returns
        -------
        MathNode
            Symbolic division node.
        """
        return MathNode.div(other, cast(Any, self))
    # end def __rtruediv__

    def __pow__(self, other: Any) -> MathNode:
        """Create a power expression with ``other``.

        Parameters
        ----------
        other : Any
            Exponent operand.

        Returns
        -------
        MathNode
            Symbolic power node.
        """
        return MathNode.pow(cast(Any, self), other)
    # end def __pow__

    def __rpow__(self, other: Any) -> MathNode:
        """Create a reverse power expression.

        Parameters
        ----------
        other : Any
            Base operand.

        Returns
        -------
        MathNode
            Symbolic power node.
        """
        return MathNode.pow(other, cast(Any, self))
    # end def __rpow__

    def __neg__(self) -> MathNode:
        """Create a negation expression.

        Returns
        -------
        MathNode
            Symbolic negation node.
        """
        return MathNode.neg(cast(Any, self))
    # end def __neg__

    def __matmul__(self, other: Any) -> MathNode:
        """Create a matrix multiplication expression.

        Parameters
        ----------
        other : Any
            Right operand.

        Returns
        -------
        MathNode
            Symbolic matmul node.
        """
        return MathNode.matmul(cast(Any, self), other)
    # end def __matmul__

    def __rmatmul__(self, other: Any) -> MathNode:
        """Create a reverse matrix multiplication expression.

        Parameters
        ----------
        other : Any
            Left operand.

        Returns
        -------
        MathNode
            Symbolic matmul node.
        """
        return MathNode.matmul(other, cast(Any, self))
    # end def __rmatmul__

    def __lt__(self, other: Any) -> MathNode:
        """Raise for unsupported ordering comparisons.

        Parameters
        ----------
        other : Any
            Unused comparison target.

        Raises
        ------
        SymbolicMathNotImplementedError
            Always, because ordering is not defined for symbolic expressions.
        """
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end def __lt__

    def __le__(self, other: Any) -> MathNode:
        """Raise for unsupported ordering comparisons.

        Parameters
        ----------
        other : Any
            Unused comparison target.

        Raises
        ------
        SymbolicMathNotImplementedError
            Always, because ordering is not defined for symbolic expressions.
        """
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end def __le__

    def __gt__(self, other: Any) -> MathNode:
        """Raise for unsupported ordering comparisons.

        Parameters
        ----------
        other : Any
            Unused comparison target.

        Raises
        ------
        SymbolicMathNotImplementedError
            Always, because ordering is not defined for symbolic expressions.
        """
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end def __gt__

    def __ge__(self, other: Any) -> MathNode:
        """Raise for unsupported ordering comparisons.

        Parameters
        ----------
        other : Any
            Unused comparison target.

        Raises
        ------
        SymbolicMathNotImplementedError
            Always, because ordering is not defined for symbolic expressions.
        """
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end def __ge__

    def __getitem__(self, item: Any) -> MathNode:
        """Create a symbolic indexing expression.

        Parameters
        ----------
        item : Any
            Index expression.

        Returns
        -------
        MathNode
            Symbolic getitem node.
        """
        return MathNode.getitem(cast(Any, self), item)
    # end def __getitem__

    # endregion OVERRIDE

    # region PRIVATE

    @staticmethod
    def _validate_body(body: Any) -> None:
        """Validate that ``body`` exposes required MathExpr members.

        Parameters
        ----------
        body : Any
            Candidate expression body.

        Raises
        ------
        TypeError
            If one or more required attributes are missing.
        """
        required_attrs = (
            "dtype",
            "shape",
            "eval",
            "variables",
            "constants",
            "contains",
            "contains_variable",
            "contains_constant",
            "is_constant",
            "is_variable",
            "diff",
        )
        missing = [name for name in required_attrs if not hasattr(body, name)]
        if missing:
            raise TypeError(
                "body must implement MathExpr members. "
                f"Missing: {', '.join(missing)}"
            )
        # end if
    # end def _validate_body

    @staticmethod
    def _normalize_look_for(look_for: Any) -> str:
        """Normalize the lookup mode across enum and string inputs.

        Parameters
        ----------
        look_for : Any
            Raw lookup selector.

        Returns
        -------
        str
            One of ``"any"``, ``"var"`` or ``"const"``.
        """
        if look_for is None:
            return "any"
        # end if

        if isinstance(look_for, str):
            lowered = look_for.lower()
            if lowered in {"variable", "var"}:
                return "var"
            # end if
            if lowered in {"constant", "const"}:
                return "const"
            # end if
            return "any"
        # end if

        name = getattr(look_for, "name", "")
        if name == "VARIABLE":
            return "var"
        # end if
        if name == "CONSTANT":
            return "const"
        # end if
        return "any"
    # end def _normalize_look_for

    def _bind_call_arguments(
            self,
            *,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind positional and keyword call arguments to variable names.

        Parameters
        ----------
        args : tuple[Any, ...]
            Positional argument values.
        kwargs : Mapping[str, Any]
            Keyword argument values.

        Returns
        -------
        dict[str, Any]
            Full variable binding map.

        Raises
        ------
        TypeError
            If argument binding is invalid.
        """
        if len(args) > len(self._variable_names):
            raise TypeError(
                f"{self.name} expects at most {len(self._variable_names)} positional arguments, "
                f"got {len(args)}."
            )
        # end if

        bindings: dict[str, Any] = {}
        for idx, value in enumerate(args):
            bindings[self._variable_names[idx]] = value
        # end for

        unknown = [name for name in kwargs if name not in self._variables_by_name]
        if unknown:
            unknown_repr = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown variable(s) for {self.name}: {unknown_repr}.")
        # end if

        duplicates = [name for name in kwargs if name in bindings]
        if duplicates:
            dup_repr = ", ".join(sorted(duplicates))
            raise TypeError(f"Variable(s) bound multiple times for {self.name}: {dup_repr}.")
        # end if

        bindings.update(kwargs)

        missing = [name for name in self._variable_names if name not in bindings]
        if missing:
            miss_repr = ", ".join(missing)
            raise TypeError(f"Missing value(s) for {self.name}: {miss_repr}.")
        # end if
        return bindings
    # end def _bind_call_arguments

    # endregion PRIVATE

# end class Function
