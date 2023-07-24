from __future__ import annotations

import numpy as np
import cv2
import os

# Mine
from __config import *
import __checker
import __image_modifier
import __parser
import __utils
import __video_player
import __grid
import __show

image_formats = global_config.image_formats
video_formats_all = global_config.video_formats_all
video_formats_tested = global_config.video_formats_tested

def play_video(image_source: str | List[ImageSource], add_frame_count:bool=True) -> None:
    """
    Play `image_source` in an interactive video player implemented entirely within cv2.

    :param image_source: A video path or ImageSource: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param add_frame_count: If True, add a frame count in the upper left corner.
    :return:
    """

    __checker.assert_type(add_frame_count, bool, "add_frame_count")
    if isinstance(image_source, str): # TODO write a check video string function
        __checker.assert_valid_video_path(image_source)
    else:
        old_value = global_config.show_max_image_amount
        global_config.show_max_image_amount = 1e6
        image_source = __parser.parse_arbitrary_image_source(image_source)
        global_config.show_max_image_amount = old_value
        if len(image_source) >= 2:
            image_source = __image_modifier.center_pad_images_into_nparray(image_source)
    if len(image_source) < 2:
        raise ValueError(f"Expected `source` to contain at least 2 images, but found `{len(image_source)=}`")

    __video_player.VideoPlayer(image_source, add_frame_count)


def _extract_frames(cap:cv2.VideoCapture, iterator:ndarray, info:dict, add_frame_count:bool, verbose:bool, frame_skipping_threshold:int) -> ndarray:
    assert len(iterator) > 1, f"The code ly on the fact that there's at least two frame indexes in `{iterator=}`"

    # Initialize return array (I think `np.empty` is the fastest way to do it,  but could be wrong)
    frame_count_goal = len(iterator)
    frames = np.empty((frame_count_goal, info["video_height"], info["video_width"], 3), dtype=np.uint8)
    if verbose:
        iterator = tqdm(iterator)

    try:
        # Extract other frames
        for i, frame_index in enumerate(iterator):
            if i == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            elif (frame_index-last_index) < frame_skipping_threshold:
                for _ in range(frame_index - last_index - 1): # read `frame_index-last_index` frames without doing anything to them
                    if not cap.read()[0]:
                        raise RuntimeError("Failed to read data while attempting to skip frames")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            was_successful, frame = cap.read()
            if not was_successful:
                raise RuntimeError("Unable to read the frame")
            if add_frame_count:
                frame = __image_modifier.draw_text_cv2(frame, str(frame_index + 1), position="upper_left_corner", deep_copy=False)
            frames[i, ...] = frame
            last_index = frame_index
    except Exception as e:
        raise RuntimeError(f"Failed reading the `{frame_index}`th frame due to error: `{e}`")

    cap.release()
    assert len(frames.shape) == 4, "Something went wrong"
    return frames[:, :, :, ::-1]  # ::-1 is a fast BGR --> RGB


def parse_video_to_images(video_path: str, keep_nth_frame: int=5, add_frame_count: bool = True, verbose: bool = True, max_frame_count_allowed:int = 500) -> Tuple[np.ndarray, dict]:
    """
    Iterate through `video_path` and extract every `keep_nth_frame` frame.

    :param keep_nth_frame: Number of frames to extract from the video.
    :param add_frame_count: If True, add a frame count in the upper left corner.
    :param verbose: If True, display progress information during video processing.
    :param video_path: Path to the video file.
    :param max_frame_count_allowed: If the combination of `video_path` and `keep_nth_frame` exceeds this limit,
                                    `keep_nth_frame` will be chosen so the amount of frames returned is `max_frame_count_allowed`.
    :returns: (1) a batch of images as a np.ndarray with shape (num_frames, H, W, 3)
              (2) a dict with video info
    """
    # Checks
    __checker.assert_types(
        [video_path, keep_nth_frame, add_frame_count, verbose],
        [str, int, bool, bool],
        ["video_path", "keep_nth_frame", "add_frame_count", "verbose"]
    )
    __checker.assert_valid_video_path(video_path)
    __checker.assert_positive_int(keep_nth_frame, variable_name="keep_nth_frame")

    # Load video
    cap, info = __utils.get_cv2_video_capture(video_path)
    iterator = np.arange(0, info["frame_count"]+1, keep_nth_frame)
    frame_count_goal = len(iterator)

    # Frame extraction
    if len(iterator) > max_frame_count_allowed:
        keep_nth_frame = np.ceil(info["frame_count"]/max_frame_count_allowed)
        iterator = np.arange(0, info["frame_count"], keep_nth_frame, dtype="int32")
        msg = f"`{frame_count_goal=}` is higher than the allowed `{max_frame_count_allowed=}`. `keep_nth_frame` will be adjusted to `{keep_nth_frame}`."
        warnings.warn(msg)

    # Frame
    if (frame_count_goal > 2_000) and (frame_count_goal > max_frame_count_allowed):
        msg = f"Suspiciously large `{frame_count_goal=}` detected. A total of `{frame_count_goal}` " \
              f"frames of shape `{(info['video_height'], info['video_width'])}` will be stored in RAM."
        warnings.warn(msg)

    # Extract frames
    if info["cv2_is_unstable"] and (max(iterator) > 10_000) and (np.max(np.abs(np.diff(iterator))) > 100):
        msg = f"cv2's frame count was unreliable, this prevents speedups during frame iteration."
        warnings.warn(msg)
    t = int(1e6) if info["cv2_is_unstable"] else 100
    frames = _extract_frames(cap, iterator, info, add_frame_count, verbose, frame_skipping_threshold=t)
    return frames, info


def parse_video_to_images_fixed_count(video_path: str, num_frames: int, add_frame_count: bool = True, verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Parse `video_path` into `num_frames` of equally spaced frames.

    :param video_path: Path to the video file.
    :param num_frames: Number of frames to extract from the video.
    :param add_frame_count: If True, add a frame count in the upper left corner.
    :param verbose: If True, display progress information during video processing.
    :returns: (1) a batch of images as a np.ndarray with shape (num_frames, H, W, 3)
              (2) a dict with video info
    """

    # Checks
    __checker.assert_types(
        [video_path, num_frames, add_frame_count, verbose],
        [str, int, bool, bool],
        ["video_path", "keep_nth_frame", "add_frame_count", "verbose"]
    )
    __checker.assert_valid_video_path(video_path)
    __checker.assert_positive_int(num_frames, variable_name="num_frames")

    # Initialize video feed
    cap, info = __utils.get_cv2_video_capture(video_path)

    # Determine which frame indexes should be extracted
    all_frame_indexes = np.arange(info["frame_count"]+1)
    iterator = __utils.extract_equally_spaced_numbers(all_frame_indexes, num_frames)

    # Extract frames
    if info["cv2_is_unstable"] and (max(iterator) > 10_000) and (np.max(np.abs(np.diff(iterator))) > 100):
        msg = f"cv2's frame count was unreliable, this prevents speedups during frame iteration."
        warnings.warn(msg)
    t = int(1e6) if info["cv2_is_unstable"] else 100
    frames = _extract_frames(cap, iterator, info, add_frame_count, verbose, frame_skipping_threshold=t)
    return frames, info


def show_video(
        video_path:str,
        num_frames:int=6,
        add_frame_count:bool=True,
        verbose:bool=True,
        add_video_details:bool=True,
        title: Optional[str] = None,
        display_image: bool = True,
        return_image: bool = False,
        max_output_image_size_wh: Optional[tuple] = (1920, 1080),
        save_image_path: Optional[str] = None,
        resize_factor: float = 1.0,
        BGR2RGB: bool = False,
) -> Optional[ndarray]:
    """
    Generate a grid image that displays `num_frames` equally spaced frames from `video_path`.

    :param video_path: Path to the video file.
    :param num_frames: Number of frames that will be displayed (they will be equally spaced across the video, but always include the first and last frame) .
    :param add_frame_count: If True, add frame count to the displayed frames. Default is True.
    :param verbose: If True, display information about the video. Default is True.
    :param add_video_details: If True, add video details (fps, frame size, etc.) to the displayed frames.
    :param title: Custom title for the displayed frames. If not provided, the video file name will be used as the title.
    :param display_image: If True, display the processed image using cv2.imshow() or PIL if in jupyter.
    :param return_image: If True, return the processed image as a np.ndarray (H, W, 3).
    :param max_output_image_size_wh: Maximum width and height of the displayed/returned image. If None, nothing will be done.
    :param save_image_path: If provided, save the processed image at the given file path. Default is None (no saving).
    :param resize_factor: Resize factor for each image/frame. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB before display/return.

    :returns: If return_image is True, return an image of shape np.ndarray (H, W, 3), otherwise None
    """

    # Checks
    __checker.assert_types(
        [video_path, num_frames, add_frame_count, verbose, add_video_details],
        [str, int, bool, bool, bool],
        ["video_path", "keep_nth_frame", "add_frame_count", "verbose", "add_path_to_title"]
    )
    __checker.assert_valid_video_path(video_path)
    __checker.assert_positive_int(num_frames, variable_name="num_frames")
    __checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, True)

    # Parse images and plot them in a grid
    images, info = parse_video_to_images_fixed_count(video_path, num_frames, add_frame_count, verbose)
    final = __grid.show_grid(
        image_source=[__parser.resize_and_bgr_parser(image, resize_factor, BGR2RGB) for image in images],
        allow_auto_resize=False,
        title=None,
        display_image=False,
        return_image=True,
        max_output_image_size_wh=max_output_image_size_wh,
        save_image_path=None,
        resize_factor=1.0,
        BGR2RGB=False,
        parse_image=False
    )

    # Title
    if add_video_details:
        video_path_abs = os.path.abspath(video_path)
        final = __image_modifier.draw_image_title(final, info["as_string"], top_padding_color=(245, 245, 245), font_size=10, text_color=(100, 100, 100), top_padding=16)
        final = __image_modifier.draw_image_title(final, video_path_abs, top_padding_color=(245,245,245), font_size=10, text_color=(100,100,100), top_padding=16, italic=True)
    if title is not None:
        final = __image_modifier.draw_image_title(final, title, top_padding_color=(245, 245, 245))
    if save_image_path is not None:
        __utils.save_image_to_disk(save_image_path, final)
    if display_image:
        __show.show_image(final)
    if return_image:
        return final

__all__ = ["play_video", "parse_video_to_images", "parse_video_to_images_fixed_count", "show_video"]