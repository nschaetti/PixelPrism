from __future__ import annotations

from .typing_expr import ExprPattern, p_any, p_const, p_node, p_var


__all__ = [
    "P",
    "PatternLibrary",
    "patterns",
]


class P:
    """Compact expression-pattern builders."""

    @staticmethod
    def n(
            op: str | None = None,
            *operands: ExprPattern,
            comm: bool = False,
            arity: int | None = None,
            var: bool | None = None,
            const: bool = False,
            shape=None,
            dtype=None,
            as_: str | None = None,
    ):
        return p_node(
            op,
            *operands,
            comm=comm,
            arity=arity,
            var=var,
            const=const,
            shape=shape,
            dtype=dtype,
            as_=as_,
        )
    # end def n

    @staticmethod
    def v(
            var_name: str | None = None,
            *,
            shape=None,
            dtype=None,
            as_: str | None = None,
    ):
        return p_var(var_name=var_name, shape=shape, dtype=dtype, as_=as_)
    # end def v

    @staticmethod
    def c(
            value=None,
            const_name: str | None = None,
            *,
            shape=None,
            dtype=None,
            as_: str | None = None,
    ):
        return p_const(value=value, const_name=const_name, shape=shape, dtype=dtype, as_=as_)
    # end def c

    @staticmethod
    def a(
            *,
            shape=None,
            dtype=None,
            as_: str | None = None,
    ):
        return p_any(shape=shape, dtype=dtype, as_=as_)
    # end def a

# end class P


class PatternLibrary:
    """Reusable classic symbolic math patterns."""

    @staticmethod
    def add(lhs: ExprPattern | None = None, rhs: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("add", lhs or P.a(), rhs or P.a(), comm=True, as_=as_)
    # end def add

    @staticmethod
    def sub(lhs: ExprPattern | None = None, rhs: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("sub", lhs or P.a(), rhs or P.a(), as_=as_)
    # end def sub

    @staticmethod
    def mul(lhs: ExprPattern | None = None, rhs: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("mul", lhs or P.a(), rhs or P.a(), comm=True, as_=as_)
    # end def mul

    @staticmethod
    def div(lhs: ExprPattern | None = None, rhs: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("div", lhs or P.a(), rhs or P.a(), as_=as_)
    # end def div

    @staticmethod
    def neg(operand: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("neg", operand or P.a(), as_=as_)
    # end def neg

    @staticmethod
    def pow(base: ExprPattern | None = None, exponent: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("pow", base or P.a(), exponent or P.a(), as_=as_)
    # end def pow

    @staticmethod
    def log(operand: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("log", operand or P.a(), as_=as_)
    # end def log

    @staticmethod
    def exp(operand: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("exp", operand or P.a(), as_=as_)
    # end def exp

    @staticmethod
    def square(operand: ExprPattern | None = None, *, as_: str | None = None):
        return P.n("pow", operand or P.a(), P.c(2.0), as_=as_)
    # end def square

    @staticmethod
    def axn(*, a_capture: str = "a_const", x_capture: str = "x_base", n_capture: str = "n_exp", as_: str | None = "mul_axn"):
        return P.n(
            "mul",
            P.c(as_=a_capture),
            P.n(
                "pow",
                P.v("x", as_=x_capture),
                P.v("n", as_=n_capture),
                as_="pow_node",
            ),
            comm=True,
            as_=as_,
        )
    # end def axn

    @staticmethod
    def ax2(*, a_capture: str = "a_const", x_capture: str = "x_base", as_: str | None = "mul_ax2"):
        return P.n(
            "mul",
            P.c(as_=a_capture),
            P.n(
                "pow",
                P.v("x", as_=x_capture),
                P.c(2.0, as_="two_exp"),
                as_="pow_node",
            ),
            comm=True,
            as_=as_,
        )
    # end def ax2

    @staticmethod
    def alogb(*, a_capture: str = "a_const", b_name: str = "b", b_capture: str = "b_const", as_: str | None = "mul_alogb"):
        return P.n(
            "mul",
            P.c(as_=a_capture),
            P.n("log", P.c(const_name=b_name, as_=b_capture), as_="log_b"),
            comm=True,
            as_=as_,
        )
    # end def alogb

    @staticmethod
    def alog_ratio(*, a_capture: str = "a_const", n_capture: str = "n_var", m_capture: str = "m_var", as_: str | None = "mul_ratio"):
        return P.n(
            "mul",
            P.c(as_=a_capture),
            P.n(
                "div",
                P.n("log", P.v("n", as_=n_capture), as_="log_n"),
                P.n("log", P.v("m", as_=m_capture), as_="log_m"),
                as_="ratio",
            ),
            comm=True,
            as_=as_,
        )
    # end def alog_ratio

# end class PatternLibrary


patterns = PatternLibrary()
