# Expressions

Expressions let you author concise formulas that control effect parameters, layer
properties, or animation channels. This guide explains the moving parts so you can
blend declarative snippets with imperative Python code.

## Expression Helpers

The :mod:`pixelprism.basic` module bundles helper functions such as ``p2``, ``s``,
and ``c`` for defining points, scalars, and colors. These helpers keep your
expressions concise and easy to serialize.

```python
from pixelprism.basic import c, p2

highlight = c(1.0, 0.8, 0.1)
cursor_path = [p2(0, 0), p2(0.25, 0.5), p2(1, 1)]
```

## Mixing Expressions with Pipelines

Attach expressions to nodes managed by :class:`pixelprism.effect_pipeline.EffectPipeline`
to re-evaluate values on every frame. Pipelines can query custom callables, easing
functions, or AI-powered analyses before handing values to downstream render passes.

## Debugging Tips

- Prefer small, pure functions so expressions stay deterministic.
- Use ``LayerManager`` to inspect layer state between evaluations.
- Log intermediate values in your :mod:`pixelprism.utils` helpers and feed them back
  into ``AnimationViewer`` for a live sanity check.
