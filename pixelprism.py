
# Imports
import argparse
import importlib.util


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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")
    parser.add_argument("--display", action="store_true", help="Display the video while processing")
    parser.add_argument("--debug-frames", type=int, nargs='*', help="List of frame numbers to debug")
    parser.add_argument("--class-file", required=True, help="Path to the file containing the CustomAnimation class")
    parser.add_argument("--class-name", required=True, help="Name of the CustomAnimation class to use")
    parser.add_argument("--kwargs", nargs='*', help="Additional keyword arguments for the CustomAnimation class in key=value format")
    args = parser.parse_args()

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

    # Create the video composer
    composer = CustomAnimationClass(
        input_path=args.input,
        output_path=args.output,
        display=args.display,
        debug_frames=args.debug_frames,
        **kwargs
    )

    # Compose the video
    composer.compose_video()
# end if
