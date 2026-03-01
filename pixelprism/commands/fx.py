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
# Copyright (C) 2026 Pixel Prism
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

"""Click command for post-processing visual effects.

This module provides the ``fx`` command, which takes an input video, applies
selected effects, and writes the processed output video.
"""

# Imports
from __future__ import annotations

from typing import Tuple

import click
import cv2
from tqdm import tqdm

from ..effect_pipeline import EffectPipeline
from ..effects.chromatic import (
    ChromaticSpatialShiftEffect,
    ChromaticTemporalPersistenceEffect,
)
from ..effects.colors import LUTEffect
from ..effects.effects import (
    AdvancedTVEffect,
    BlurEffect,
    GlowEffect,
    LenticularDistortionEffect,
)
from ..effects.interest_points import SIFTPointsEffect
from ..layer_manager import LayerManager


def _str_to_tuple(value: str) -> Tuple[int, int]:
    """Convert a ``x,y`` string to a 2D integer tuple.

    Parameters
    ----------
    value : str
        Input string in ``x,y`` format.

    Returns
    -------
    tuple[int, int]
        Parsed 2D integer tuple.

    Raises
    ------
    click.BadParameter
        If ``value`` cannot be parsed as two integers.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise click.BadParameter("Expected format 'x,y'.")
    # end if

    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise click.BadParameter("Shift values must be integers.") from exc
    # end try
# end def _str_to_tuple


def _parse_border_color(value: str) -> Tuple[int, int, int]:
    """Convert a ``b,g,r`` string to a 3D integer tuple.

    Parameters
    ----------
    value : str
        Border color encoded as ``b,g,r``.

    Returns
    -------
    tuple[int, int, int]
        Parsed color tuple.

    Raises
    ------
    click.BadParameter
        If ``value`` does not contain three integer channels.
    """
    parts = value.split(",")
    if len(parts) != 3:
        raise click.BadParameter("Expected border color format 'b,g,r'.")
    # end if

    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise click.BadParameter("Border color channels must be integers.") from exc
    # end try
# end def _parse_border_color


def process_video(
    input_path: str,
    output_path: str,
    effect_pipeline: EffectPipeline,
    fps_modifier: float = 1.0,
) -> None:
    """Apply effects frame-by-frame on an input video.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    output_path : str
        Path to the output video file.
    effect_pipeline : EffectPipeline
        Pipeline used to process each frame.
    fps_modifier : float, default=1.0
        Value used to adjust output FPS.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
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

    with tqdm(total=frame_count, desc="Processing video") as progress:
        ret, frame = cap.read()

        while cap.isOpened():
            if not ret:
                break
            # end if

            combined_effect = effect_pipeline.apply(frame)
            out.write(combined_effect)  # pyright: ignore[reportCallIssue, reportArgumentType]

            ret, frame = cap.read()
            progress.update(1)
        # end while
    # end with

    cap.release()
    out.release()
    effect_pipeline.print_stats()
# end def process_video


@click.command("fx", help="Apply visual effects to an existing input video.")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--enable_tv_effect", is_flag=True, help="Enable TV effect.")
@click.option("--pixel_width", type=int, default=10, show_default=True, help="Pixel width for TV effect.")
@click.option("--pixel_height", type=int, default=10, show_default=True, help="Pixel height for TV effect.")
@click.option("--border_strength", type=int, default=2, show_default=True, help="Border strength for TV effect.")
@click.option("--border_color", type=str, default="0,0,0", show_default=True, help="Border color in BGR format.")
@click.option("--corner_radius", type=int, default=2, show_default=True, help="Radius for corner clipping.")
@click.option("--blur_kernel_size", type=int, default=5, show_default=True, help="Gaussian blur kernel size for overlay.")
@click.option("--vertical_shift", type=int, default=0, show_default=True, help="Vertical shift for column rectangles.")
@click.option("--enable_chromatic_shift", is_flag=True, help="Enable chromatic spatial shift effect.")
@click.option("--shift_r", type=str, default="5,0", show_default=True, help="Red channel shift as x,y.")
@click.option("--shift_g", type=str, default="-5,0", show_default=True, help="Green channel shift as x,y.")
@click.option("--shift_b", type=str, default="0,5", show_default=True, help="Blue channel shift as x,y.")
@click.option("--enable_chromatic_persistence", is_flag=True, help="Enable chromatic temporal persistence effect.")
@click.option("--persistence_r", type=int, default=5, show_default=True, help="Persistence for red channel.")
@click.option("--persistence_g", type=int, default=5, show_default=True, help="Persistence for green channel.")
@click.option("--persistence_b", type=int, default=5, show_default=True, help="Persistence for blue channel.")
@click.option("--enable_lenticular_distortion", is_flag=True, help="Enable lenticular distortion effect.")
@click.option("--distortion_strength", type=float, default=0.00001, show_default=True, help="Lenticular distortion strength.")
@click.option("--enable_glow", is_flag=True, help="Enable glow effect.")
@click.option("--glow_blur_strength", type=int, default=5, show_default=True, help="Glow blur strength.")
@click.option("--glow_intensity", type=float, default=0.5, show_default=True, help="Glow intensity.")
@click.option(
    "--glow_blend_mode",
    type=click.Choice(["addition", "multiply", "screen", "overlay"]),
    default="screen",
    show_default=True,
    help="Glow blend mode.",
)
@click.option("--enable_blur", is_flag=True, help="Enable blur effect.")
@click.option("--blur_strength", type=int, default=5, show_default=True, help="Blur strength.")
@click.option("--enable_sift", is_flag=True, help="Enable SIFT points effect.")
@click.option("--sift_num_octaves", type=int, default=4, show_default=True, help="Number of SIFT octaves.")
@click.option("--sift_num_scales", type=int, default=3, show_default=True, help="Number of scales per SIFT octave.")
@click.option("--enable_lut", is_flag=True, help="Enable LUT effect.")
@click.option("--lut_path", type=click.Path(exists=True), default=None, help="Path to LUT file.")
def fx(
    input: str,
    output: str,
    enable_tv_effect: bool,
    pixel_width: int,
    pixel_height: int,
    border_strength: int,
    border_color: str,
    corner_radius: int,
    blur_kernel_size: int,
    vertical_shift: int,
    enable_chromatic_shift: bool,
    shift_r: str,
    shift_g: str,
    shift_b: str,
    enable_chromatic_persistence: bool,
    persistence_r: int,
    persistence_g: int,
    persistence_b: int,
    enable_lenticular_distortion: bool,
    distortion_strength: float,
    enable_glow: bool,
    glow_blur_strength: int,
    glow_intensity: float,
    glow_blend_mode: str,
    enable_blur: bool,
    blur_strength: int,
    enable_sift: bool,
    sift_num_octaves: int,
    sift_num_scales: int,
    enable_lut: bool,
    lut_path: str | None,
) -> None:
    """Execute the effects pipeline command.

    Parameters
    ----------
    input : str
        Path to input video.
    output : str
        Path to output video.
    enable_tv_effect : bool
        Enable CRT-like TV effect.
    pixel_width : int
        TV effect pixel width.
    pixel_height : int
        TV effect pixel height.
    border_strength : int
        TV effect border thickness.
    border_color : str
        TV effect border color encoded as ``b,g,r``.
    corner_radius : int
        TV effect corner clipping radius.
    blur_kernel_size : int
        TV effect overlay blur kernel size.
    vertical_shift : int
        Vertical shift for TV effect columns.
    enable_chromatic_shift : bool
        Enable spatial chromatic shift.
    shift_r : str
        Red channel shift string.
    shift_g : str
        Green channel shift string.
    shift_b : str
        Blue channel shift string.
    enable_chromatic_persistence : bool
        Enable temporal chromatic persistence.
    persistence_r : int
        Red channel persistence.
    persistence_g : int
        Green channel persistence.
    persistence_b : int
        Blue channel persistence.
    enable_lenticular_distortion : bool
        Enable lenticular distortion.
    distortion_strength : float
        Distortion strength.
    enable_glow : bool
        Enable glow effect.
    glow_blur_strength : int
        Glow blur strength.
    glow_intensity : float
        Glow intensity.
    glow_blend_mode : str
        Glow blend mode.
    enable_blur : bool
        Enable blur effect.
    blur_strength : int
        Blur effect strength.
    enable_sift : bool
        Enable SIFT points effect.
    sift_num_octaves : int
        Number of SIFT octaves.
    sift_num_scales : int
        Number of scales per octave.
    enable_lut : bool
        Enable LUT effect.
    lut_path : str | None
        Optional LUT path.
    """
    border_color_tuple = _parse_border_color(border_color)
    shift_r_tuple = _str_to_tuple(shift_r)
    shift_g_tuple = _str_to_tuple(shift_g)
    shift_b_tuple = _str_to_tuple(shift_b)

    effects = []

    layer_manager = LayerManager()
    layer_manager.add_layer("input_layer")

    if enable_tv_effect:
        effects.append(
            AdvancedTVEffect(
                pixel_width,
                pixel_height,
                border_strength,
                border_color_tuple,
                corner_radius,
                blur_kernel_size,
                vertical_shift,
            )
        )
    # end if

    if enable_chromatic_shift:
        click.echo("Enabling Chromatic Spatial Shift effect")
        effects.append(
            ChromaticSpatialShiftEffect(shift_r_tuple, shift_g_tuple, shift_b_tuple)
        )
    # end if

    if enable_chromatic_persistence:
        click.echo("Enabling Chromatic Temporal Persistence effect")
        effects.append(
            ChromaticTemporalPersistenceEffect(
                persistence_r,
                persistence_g,
                persistence_b,
            )
        )
    # end if

    if enable_lenticular_distortion:
        click.echo("Enabling Lenticular Distortion effect")
        effects.append(LenticularDistortionEffect(distortion_strength))
    # end if

    if enable_glow:
        click.echo("Enabling Glow effect")
        effects.append(
            GlowEffect(
                glow_blur_strength,
                glow_intensity,
                glow_blend_mode,
            )
        )
    # end if

    if enable_blur:
        click.echo("Enabling Blur effect")
        effects.append(BlurEffect(blur_strength))
    # end if

    if enable_sift:
        click.echo("Enabling SIFT Points effect")
        effects.append(SIFTPointsEffect(sift_num_octaves, sift_num_scales))
    # end if

    if enable_lut and lut_path:
        click.echo("Enabling LUT effect")
        effects.append(LUTEffect(lut_path))
    # end if

    effect_pipeline = EffectPipeline(effects)
    process_video(input, output, effect_pipeline)
# end def fx
