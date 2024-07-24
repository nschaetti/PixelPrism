
# Imports
import argparse
import importlib.util

from pixel_prism import VideoComposer
from pixel_prism.utils import setup_logger


# Setup logger
logger = setup_logger(__name__)


def load_class_from_file(
        file_path,
        class_name
):
    """
    Load a class from a file.

    Args:
        file_path (str): Path to the file containing the class
        class_name (str): Name of the class to load
    """
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
# end load_class_from_file


# Boolean argument
def bool_arg(value):
    """
    Parse a boolean argument from the command line.

    Args:
        value (str): Value to parse

    Returns:
        bool: Parsed boolean value
    """
    if type(value) is bool:
        return value
    # end if
    return value.lower() in ("yes", "true", "t", "1")
# end bool_arg


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("--input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")
    parser.add_argument("--display", action="store_true", help="Display the video while processing")
    parser.add_argument("--debug-frames", type=int, nargs='*', help="List of frame numbers to debug")
    parser.add_argument("--class-file", required=True, help="Path to the file containing the CustomAnimation class")
    parser.add_argument("--class-name", required=True, help="Name of the CustomAnimation class to use")
    parser.add_argument("--duration", type=float, help="Duration of the animation in seconds.")
    parser.add_argument("--fps", type=int, help="Frames per second of the animation.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output video.")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output video.")
    parser.add_argument("--save-frames", type=bool_arg, help="Save the frames to disk.")
    parser.add_argument("--kwargs", nargs='*', help="Additional keyword arguments for the CustomAnimation class in key=value format")
    args = parser.parse_args()

    # Show version
    logger.info("PixelPrism v0.1.0")

    # Check that duration is provided if no input video is given
    if not args.input and not args.duration:
        parser.error("Duration (--duration) is required if no input video is specified.")
    # end if

    # Load the CustomAnimation class from the specified file
    CustomAnimationClass = load_class_from_file(args.class_file, args.class_name)

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            key, value = kwarg.split('=')
            kwargs[key] = value
        # end for
    # end if

    # Log additional parameters
    logger.info(f"CustomAnimation class: {args.class_name}")
    logger.info(f"Duration: {args.duration}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Width: {args.width}")
    logger.info(f"Height: {args.height}")
    logger.info(f"Save frames: {args.save_frames}")
    logger.info(f"Keyword arguments: {kwargs}")

    # Create video composer
    composer = VideoComposer(
        input_path=args.input,
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        animation_class=CustomAnimationClass,
        debug_frames=args.debug_frames,
        save_frames=args.save_frames,
        **kwargs
    )

    # Create video
    composer.create_video()
# end if
