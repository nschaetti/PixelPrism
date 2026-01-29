# Animation

Pixel Prism animations combine timeline-aware helpers with rendering backends so you
can script motion graphics directly in Python.

## Core Classes

- ``pixelprism.animation.Animation`` – Base timeline controller that steps frames,
  interpolates parameters, and calls your ``render_frame`` hooks.
- ``pixelprism.animation.AnimationViewer`` – Lightweight Qt-based viewer for quick
  previews while tuning parameters.
- ``pixelprism.video_composer.VideoComposer`` – Streams rendered frames to high
  quality video files with audio support.

## Building an Animation

```python
from pixelprism.animation import Animation
from pixelprism.effect_pipeline import EffectPipeline

class Orbit(Animation):
    def setup(self):
        self.pipeline = EffectPipeline()
        # register nodes, load textures, etc.

    def render_frame(self, frame, t):
        self.pipeline["orbit"].angle = t * 0.5
        return self.pipeline.render()

animation = Orbit(duration=5.0, fps=30)
animation.render("orbit.mp4")
```

## Best Practices

- Keep expensive state inside pipelines or render engines and reuse them each frame.
- Separate animation math into mixins (see ``pixelprism.mixins``) to share behaviors
  across multiple scenes.
- Pair ``VideoComposer`` with ``RenderEngine`` when exporting UHD or HDR results.
