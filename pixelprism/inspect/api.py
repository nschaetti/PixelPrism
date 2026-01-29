"""
Public inspection façade for math expressions.
"""

from __future__ import annotations

from typing import Callable, Dict

from pixelprism.math import MathNode

from .extract import Graph, extract


Renderer = Callable[[Graph], object]

_RENDERERS: Dict[str, Callable[..., object]] = {}


def _loader_console():
    from . import console_tree
    return console_tree.render
# end def _loader_console

def _loader_topological():
    from . import topological
    return topological.render
# end def _loader_topological

def _loader_sexpr():
    from . import sexpr
    return sexpr.render
# end def _loader_sexpr

def _loader_graphviz():
    from . import graphviz
    return graphviz.render
# end def _loader_graphviz


_BACKENDS: Dict[str, Callable[[], Callable[..., object]]] = {
    "console": _loader_console,
    "topological": _loader_topological,
    "sexpr": _loader_sexpr,
    "graphviz": _loader_graphviz,
}


def inspect(
        expr: MathNode,
        *,
        backend: str = "topological",
        **backend_kwargs
) -> object:
    """
    Inspect a MathExpr using the selected backend.

    Parameters
    ----------
    expr:
        The expression to inspect.
    backend:
        Name of the rendering backend to use.
    backend_kwargs:
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    object
        Backend-specific rendering result.
    """
    graph = extract(expr)
    if backend not in _RENDERERS:
        try:
            loader = _BACKENDS[backend]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown inspection backend '{backend}'") from exc
        # end try
        _RENDERERS[backend] = loader()
    # end if
    renderer = _RENDERERS[backend]
    return renderer(graph, **backend_kwargs)
# end def inspect
