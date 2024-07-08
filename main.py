import cv2
import numpy as np
from tqdm import tqdm
import time
import argparse

from pixel_prism.effect_pipeline import EffectPipeline
from pixel_prism.effects.effects import AdvancedTVEffect, ChromaticAberrationEffect, LenticularDistortionEffect
from pixel_prism.effects.effects import GlowEffect, BlurEffect
from pixel_prism.effects.chromatic import ChromaticSpatialShiftEffect, ChromaticTemporalPersistenceEffect
from pixel_prism.effects.interest_points import SIFTPointsEffect
from pixel_prism.effects.colors import LUTEffect


def process_video(
        input_path,
        output_path,
        effect_pipeline,
        fps_modifier=1
):
    """
    Process a video file with the specified effect pipeline and save the output to a new file.

    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the output video file
        effect_pipeline (EffectPipeline): Effect pipeline to apply to the video
        fps_modifier (float): Modifier to adjust the output FPS
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) / fps_modifier
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

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
# end process_video


def str_to_tuple(s):
    """
    Convert a string to a tuple of integers.

    Args:
        s (str): String to convert to tuple
    """
    return tuple(map(int, s.split(',')))
# end str_to_tuple


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")

    # Add arguments for TV effect
    parser.add_argument("--enable_tv_effect", action='store_true', help="Enable TV effect")
    parser.add_argument("--pixel_width", type=int, default=10, help="Pixel width for TV effect")
    parser.add_argument("--pixel_height", type=int, default=10, help="Pixel height for TV effect")
    parser.add_argument("--border_strength", type=int, default=2, help="Border strength for TV effect")
    parser.add_argument("--border_color", type=str, default="0,0,0", help="Border color for TV effect in BGR format")
    parser.add_argument("--corner_radius", type=int, default=2, help="Radius for cutting the corners in TV effect")
    parser.add_argument("--blur_kernel_size", type=int, default=5, help="Kernel size for Gaussian blur applied to the overlay")
    parser.add_argument("--vertical_shift", type=int, default=0, help="Vertical shift for rectangles in each column")

    # Add arguments for Chromatic Aberration effect
    parser.add_argument("--enable_chromatic_shift", action='store_true', help="Enable Chromatic Spatial Shift effect")
    parser.add_argument("--shift_r", type=str_to_tuple, default="5,0", help="Shift for the red channel in chromatic spatial shift")
    parser.add_argument("--shift_g", type=str_to_tuple, default="-5,0", help="Shift for the green channel in chromatic spatial shift")
    parser.add_argument("--shift_b", type=str_to_tuple, default="0,5", help="Shift for the blue channel in chromatic spatial shift")

    # Add arguments for Chromatic Temporal Persistence effect
    parser.add_argument("--enable_chromatic_persistence", action='store_true', help="Enable Chromatic Temporal Persistence effect")
    parser.add_argument("--persistence_r", type=int, default=5, help="Persistence for red channel in chromatic temporal effect")
    parser.add_argument("--persistence_g", type=int, default=5, help="Persistence for green channel in chromatic temporal effect")
    parser.add_argument("--persistence_b", type=int, default=5, help="Persistence for blue channel in chromatic temporal effect")

    # Add arguments for Lenticular Distortion effect
    parser.add_argument("--enable_lenticular_distortion", action='store_true', help="Enable Lenticular Distortion effect")
    parser.add_argument("--distortion_strength", type=float, default=0.00001, help="Strength of the lenticular distortion effect")

    # Add arguments for Glow effect
    parser.add_argument("--enable_glow", action='store_true', help="Enable Glow effect")
    parser.add_argument("--glow_blur_strength", type=int, default=5, help="Blur strength for glow effect")
    parser.add_argument("--glow_intensity", type=float, default=0.5, help="Intensity of the glow effect")
    parser.add_argument("--glow_blend_mode", type=str, default='screen', choices=['addition', 'multiply', 'screen', 'overlay'], help="Blend mode for the glow effect")

    # Add arguments for Blur effect
    parser.add_argument("--enable_blur", action='store_true', help="Enable Blur effect")
    parser.add_argument("--blur_strength", type=int, default=5, help="Blur strength for blur effect")

    # Add arguments for SIFT Points effect
    parser.add_argument("--enable_sift", action='store_true', help="Enable SIFT Points effect")
    parser.add_argument("--sift_num_octaves", type=int, default=4, help="Number of octaves for SIFT")
    parser.add_argument("--sift_num_scales", type=int, default=3, help="Number of scales per octave for SIFT")

    # Add arguments for LUT effect
    parser.add_argument("--enable_lut", action='store_true', help="Enable LUT effect")
    parser.add_argument("--lut_path", type=str, help="Path to the LUT file")

    # Parse arguments
    args = parser.parse_args()

    # Parse the arguments for the TV effect
    border_color = tuple(map(int, args.border_color.split(',')))

    effects = []

    layer_manager = LayerManager()
    layer_manager.add_layer("input_layer")

    # Parse the arguments for the TV effect
    if args.enable_tv_effect:
        effects.append(
            AdvancedTVEffect(
                args.pixel_width,
                args.pixel_height,
                args.border_strength,
                border_color,
                args.corner_radius,
                args.blur_kernel_size,
                args.vertical_shift)
        )
    # end if

    # Parse the arguments for the Chromatic Aberration effect
    if args.enable_chromatic_shift:
        print(f"Enabling Chromatic Spatial Shift effect")
        effects.append(ChromaticSpatialShiftEffect(args.shift_r, args.shift_g, args.shift_b))
    # end if

    # Parse the arguments for the Chromatic Temporal Persistence effect
    if args.enable_chromatic_persistence:
        print(f"Enabling Chromatic Temporal Persistence effect")
        effects.append(ChromaticTemporalPersistenceEffect(args.persistence_r, args.persistence_g, args.persistence_b))
    # end if

    # Parse the arguments for the Lenticular Distortion effect
    if args.enable_lenticular_distortion:
        print(f"Enabling Lenticular Distortion effect")
        effects.append(LenticularDistortionEffect(args.distortion_strength))
    # end if

    # Parse the arguments for the Glow effect
    if args.enable_glow:
        print(f"Enabling Glow effect")
        effects.append(GlowEffect(args.glow_blur_strength, args.glow_intensity, args.glow_blend_mode))
    # end if

    # Parse the arguments for the Blur effect
    if args.enable_blur:
        print(f"Enabling Blur effect")
        effects.append(BlurEffect(args.blur_strength))
    # end if

    # Parse the arguments for the SIFT Points effect
    if args.enable_sift:
        print(f"Enabling SIFT Points effect")
        effects.append(SIFTPointsEffect(args.sift_num_octaves, args.sift_num_scales))
    # end if

    # Parse the arguments for the LUT effect
    if args.enable_lut and args.lut_path:
        print(f"Enabling LUT effect")
        effects.append(LUTEffect(args.lut_path))
    # end if

    # Create an effect pipeline with the specified effects
    effect_pipeline = EffectPipeline(effects)

    # Process the video with the effect pipeline
    process_video(args.input, args.output, effect_pipeline)
# end main

