# Overview

Pixel Prism is designed around small, composable building blocks that can be glued
together to create larger visual systems. This overview explains the core projects
pieces you will encounter while browsing the rest of the guides.

## Architecture

- **Nodes and Pipelines** – Use ``pixelprism.effect_pipeline.EffectPipeline`` to build
  ordered chains of effects, compositing passes, and custom Python callbacks.
- **Animation Primitives** – ``pixelprism.animation.Animation`` drives frame updates
  while ``pixelprism.animation.AnimationViewer`` handles preview rendering.
- **Rendering Layer** – ``pixelprism.render_engine.RenderEngine`` exposes the low level
  routines that talk to image buffers, depth maps, and GPU shaders.

## Typical Workflow

1. Create or load media assets (images, depth maps, video).
2. Compose an effect pipeline that mixes stock nodes with custom operations.
3. Animate parameters or nodes using ``Animation`` helpers or keyframe mixins.
4. Render the result either to disk or to an interactive viewer.

## What to Read Next

- Continue with the [Expressions](expressions.md) guide to script parameter changes.
- Jump to [Animation](animation.md) for time-based control.
- Explore [Math Sampling](math_sampling.md), [Math Statistics](math_statistics.md),
  [Math Graphs](math_graphs.md), and [Math Machine Learning](math_machine_learning.md)
  for symbolic numeric workflows.
- Visit the :doc:`../api/modules` reference for in-depth API documentation.
