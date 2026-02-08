"""
Optional caches for LaTeX conversion results.

The cache is deliberately opt-in to keep rendering functions pure and free of
implicit global state.
"""

from __future__ import annotations

import weakref
from typing import Callable

from ..math_base import MathNode

__all__ = ["LatexRenderCache"]


class LatexRenderCache:
    """
    Cache LaTeX strings keyed by :class:`MathExpr` identity.

    Notes
    -----
    The cache is implemented with :class:`weakref.WeakKeyDictionary` so that
    entries automatically disappear once the expression is no longer referenced
    elsewhere. Use it as a lightweight memoization helper when rendering the
    same tree repeatedly.
    """

    def __init__(self) -> None:
        self._store: "weakref.WeakKeyDictionary[MathNode, str]" = weakref.WeakKeyDictionary()

    def get(self, expr: MathNode, factory: Callable[[MathNode], str]) -> str:
        """
        Retrieve the cached LaTeX string for ``expr`` or compute it on-demand.

        Parameters
        ----------
        expr :
            Expression tree node whose LaTeX representation is requested.
        factory :
            Callable invoked when ``expr`` is not cached. The callable is
            responsible for producing the LaTeX string.

        Returns
        -------
        str
            Cached or freshly computed LaTeX math string.
        """
        try:
            return self._store[expr]
        except KeyError:
            latex = factory(expr)
            self._store[expr] = latex
            return latex

    def clear(self) -> None:
        """
        Clear all cached entries.
        """
        self._store.clear()
