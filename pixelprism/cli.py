"""Command-line interface for the Pixel Prism video processor."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

from .effect_pipeline import EffectPipeline
from .effects.chromatic import (
    ChromaticSpatialShiftEffect,
    ChromaticTemporalPersistenceEffect,
)
from .effects.colors import LUTEffect
from .effects.effects import (
    AdvancedTVEffect,
    BlurEffect,
    GlowEffect,
    LenticularDistortionEffect,
)
from .effects.interest_points import SIFTPointsEffect
from .layer_manager import LayerManager


def str_to_tuple(value: str) -> Tuple[int, int]:
    """
    Convert a ``"x,y"`` string to a tuple of integers.

    Args:
        value: String representation of a 2D shift vector.

    Returns:
        Tuple[int, int]: Parsed shift tuple.
    """
    return tuple(map(int, value.split(",")))


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Apply visual effects to videos using Pixel Prism."
    )
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")

    # TV effect
    parser.add_argument(
        "--enable_tv_effect", action="store_true", help="Enable TV effect"
    )
    parser.add_argument(
        "--pixel_width", type=int, default=10, help="Pixel width for TV effect"
    )
    parser.add_argument(
        "--pixel_height", type=int, default=10, help="Pixel height for TV effect"
    )
    parser.add_argument(
        "--border_strength", type=int, default=2, help="Border strength for TV effect"
    )
    parser.add_argument(
        "--border_color",
        type=str,
        default="0,0,0",
        help="Border color for TV effect in BGR format",
    )
    parser.add_argument(
        "--corner_radius", type=int, default=2, help="Radius for cutting the corners"
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=5,
        help="Kernel size for Gaussian blur applied to the overlay",
    )
    parser.add_argument(
        "--vertical_shift",
        type=int,
        default=0,
        help="Vertical shift for rectangles in each column",
    )

    # Chromatic spatial shift
    parser.add_argument(
        "--enable_chromatic_shift",
        action="store_true",
        help="Enable Chromatic Spatial Shift effect",
    )
    parser.add_argument(
        "--shift_r",
        type=str_to_tuple,
        default=str_to_tuple("5,0"),
        help="Shift for the red channel in chromatic spatial shift",
    )
    parser.add_argument(
        "--shift_g",
        type=str_to_tuple,
        default=str_to_tuple("-5,0"),
        help="Shift for the green channel in chromatic spatial shift",
    )
    parser.add_argument(
        "--shift_b",
        type=str_to_tuple,
        default=str_to_tuple("0,5"),
        help="Shift for the blue channel in chromatic spatial shift",
    )

    # Chromatic temporal persistence
    parser.add_argument(
        "--enable_chromatic_persistence",
        action="store_true",
        help="Enable Chromatic Temporal Persistence effect",
    )
    parser.add_argument(
        "--persistence_r", type=int, default=5, help="Persistence for red channel"
    )
    parser.add_argument(
        "--persistence_g", type=int, default=5, help="Persistence for green channel"
    )
    parser.add_argument(
        "--persistence_b", type=int, default=5, help="Persistence for blue channel"
    )

    # Lenticular distortion
    parser.add_argument(
        "--enable_lenticular_distortion",
        action="store_true",
        help="Enable Lenticular Distortion effect",
    )
    parser.add_argument(
        "--distortion_strength",
        type=float,
        default=0.00001,
        help="Strength of the lenticular distortion effect",
    )

    # Glow
    parser.add_argument("--enable_glow", action="store_true", help="Enable Glow effect")
    parser.add_argument(
        "--glow_blur_strength",
        type=int,
        default=5,
        help="Blur strength for glow effect",
    )
    parser.add_argument(
        "--glow_intensity", type=float, default=0.5, help="Intensity of the glow effect"
    )
    parser.add_argument(
        "--glow_blend_mode",
        type=str,
        default="screen",
        choices=["addition", "multiply", "screen", "overlay"],
        help="Blend mode for the glow effect",
    )

    # Blur
    parser.add_argument("--enable_blur", action="store_true", help="Enable Blur effect")
    parser.add_argument(
        "--blur_strength", type=int, default=5, help="Blur strength for blur effect"
    )

    # SIFT
    parser.add_argument(
        "--enable_sift", action="store_true", help="Enable SIFT Points effect"
    )
    parser.add_argument(
        "--sift_num_octaves", type=int, default=4, help="Number of octaves for SIFT"
    )
    parser.add_argument(
        "--sift_num_scales",
        type=int,
        default=3,
        help="Number of scales per octave for SIFT",
    )

    # LUT
    parser.add_argument("--enable_lut", action="store_true", help="Enable LUT effect")
    parser.add_argument("--lut_path", type=str, help="Path to the LUT file")

    return parser


def process_video(
    input_path: str,
    output_path: str,
    effect_pipeline: EffectPipeline,
    fps_modifier: float = 1,
) -> None:
    """
    Process a video file with the specified effect pipeline and save the output.

    Args:
        input_path: Path to the input video file.
        output_path: Path to save the output video file.
        effect_pipeline: Effect pipeline to apply to the video.
        fps_modifier: Modifier to adjust the output FPS.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) / fps_modifier
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc="Processing video") as pbar:
        ret, frame = cap.read()

        while cap.isOpened():
            if not ret:
                break
            combined_effect = effect_pipeline.apply(frame)
            out.write(combined_effect)

            ret, frame = cap.read()
            pbar.update(1)

    cap.release()
    out.release()
    effect_pipeline.print_stats()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    border_color = tuple(map(int, args.border_color.split(",")))
    effects = []

    layer_manager = LayerManager()
    layer_manager.add_layer("input_layer")

    if args.enable_tv_effect:
        effects.append(
            AdvancedTVEffect(
                args.pixel_width,
                args.pixel_height,
                args.border_strength,
                border_color,
                args.corner_radius,
                args.blur_kernel_size,
                args.vertical_shift,
            )
        )

    if args.enable_chromatic_shift:
        print("Enabling Chromatic Spatial Shift effect")
        effects.append(
            ChromaticSpatialShiftEffect(args.shift_r, args.shift_g, args.shift_b)
        )

    if args.enable_chromatic_persistence:
        print("Enabling Chromatic Temporal Persistence effect")
        effects.append(
            ChromaticTemporalPersistenceEffect(
                args.persistence_r,
                args.persistence_g,
                args.persistence_b,
            )
        )

    if args.enable_lenticular_distortion:
        print("Enabling Lenticular Distortion effect")
        effects.append(LenticularDistortionEffect(args.distortion_strength))

    if args.enable_glow:
        print("Enabling Glow effect")
        effects.append(
            GlowEffect(
                args.glow_blur_strength, args.glow_intensity, args.glow_blend_mode
            )
        )

    if args.enable_blur:
        print("Enabling Blur effect")
        effects.append(BlurEffect(args.blur_strength))

    if args.enable_sift:
        print("Enabling SIFT Points effect")
        effects.append(SIFTPointsEffect(args.sift_num_octaves, args.sift_num_scales))

    if args.enable_lut and args.lut_path:
        print("Enabling LUT effect")
        effects.append(LUTEffect(args.lut_path))

    effect_pipeline = EffectPipeline(effects)
    process_video(args.input, args.output, effect_pipeline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
