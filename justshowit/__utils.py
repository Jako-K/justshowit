from __future__ import annotations
import os
from PIL import ImageFont
import cv2
import numpy as np
from datetime import datetime
from os.path import getmtime

from __config import *
import __checker


def save_image_to_disk(path:str, image:ndarray) -> None:
    __checker.assert_types([path, image], [str, ndarray], ["path", "image"])
    if not any(path.lower().endswith(f) for f in global_config.image_formats):
        raise ValueError(f"`{path=}` must be suffixed by: `{global_config.image_formats}`, but found neither.")
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def get_font(font_size:int, font_thickness:int, italic:bool=False):
    __checker.assert_types([font_size, font_thickness, italic], [int, int, bool], ["font_size", "font_thickness", "italic"])
    __checker.assert_positive_int(font_size, variable_name="font_size")
    if not (0 < font_thickness <= 7):
        raise ValueError(f"`{font_thickness=}` is not valid. As of writing this, `font_thickness` must be in [1,2,3,4,5,6,7]")

    # TODO: add return types
    # Maps from (font_thickness, is_italic) to font
    font_name_mapper = {
        (1, False): 'JetBrainsMono-Thin.ttf',
        (1, True):  'JetBrainsMono-ThinItalic.ttf',
        (2, False): 'JetBrainsMono-ExtraLight.ttf',
        (2, True):  'JetBrainsMono-ExtraLightItalic.ttf',
        (3, False): 'JetBrainsMono-Light.ttf',
        (3, True):  'JetBrainsMono-LightItalic.ttf',
        (4, False): 'JetBrainsMono-Medium.ttf',
        (4, True):  'JetBrainsMono-MediumItalic.ttf',
        (5, False): 'JetBrainsMono-SemiBold.ttf',
        (5, True):  'JetBrainsMono-SemiBoldItalic.ttf',
        (6, False): 'JetBrainsMono-Bold.ttf',
        (6, True):  'JetBrainsMono-BoldItalic.ttf',
        (7, False): 'JetBrainsMono-ExtraBold.ttf',
        (7, True):  'JetBrainsMono-ExtraBoldItalic.ttf'
    }

    # Load font from path
    font_name = font_name_mapper[(font_thickness, italic)]
    current_folder_path = os.path.dirname(__file__)
    absolute_font_path = os.path.join(current_folder_path, "__fonts", font_name) # This is probably overkill, but I don't want to deal with any relative path shenanigans
    if os.path.exists(absolute_font_path):
        try:
            font = ImageFont.truetype(absolute_font_path, size=font_size)
            return font
        except Exception as e:
            msg = f"Failed to load `{font_name=}` from path: `{absolute_font_path}`.\n" \
            f"This should not be possible because all __fonts are validated before reaching this point. " \
            f"So, I'm uncertain why the font cannot be located..\n" \
            f"The error:\n`{e}`"
            raise RuntimeError(msg)
    # TODO: Should there be a third option here that tries to download the font directly from the github page?
    # TODO: Find out if it makes better sense to try the encoding first and then the path
    # Load font directly from encodings
    else:
        from __encoded_font_data import fonts_encoded
        import base64
        import io
        try:
            encoded_font_data = fonts_encoded[font_name]
            binary = base64.b64decode(encoded_font_data)  # decode from base64 to binary
            FileLike = io.BytesIO(binary)  # wrap in BytesIO to make file-like object
            font = ImageFont.truetype(FileLike, size=font_size)  # load font
            return font
        except Exception as e:
            msg = f"Failed to load `{font_name=}` from path: `{absolute_font_path}` and from pre-made encodings.\n" \
                  f"This should not be possible because all __fonts are validated before reaching this point. " \
                  f"So, I'm uncertain why the font cannot be located..\n" \
                  f"The error:\n`{e}`"
            raise RuntimeError(msg)


def frame_can_be_read(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    successful, _ = cap.read()
    if successful:
        return True
    return False


def is_valid_end_frame(cap, index):
    return frame_can_be_read(cap, index) and (not frame_can_be_read(cap, index+1))


def get_cv2_video_capture(video_path: str, max_iteration: int = 50) -> Tuple[cv2.VideoCapture, dict]:
    """
    "Ensure successful loading of cv2.VideoCapture(video_path) and provide some video information, including a reliable frame count."

    :param video_path: Path to a video file
    :param max_iteration: If the frame count provided by cv2 is unreliable, binary search will be utilized to determine the frame count with a maximum number of iterations specified by max_iteration.
    :return: Successfully loaded cv2.VideoCapture and a dict with video information
    """

    __checker.assert_path(video_path, "video_path")
    __checker.assert_positive_int(max_iteration, variable_name="max_iteration")

    # -----------------------------------
    # Setup
    # -----------------------------------

    # Load and check video stream
    try:
        cap = cv2.VideoCapture(video_path)
        successful, _ = cap.read()
        if not successful:
            raise RuntimeError("Failed to load the first frame from `cv2.VideoCapture(video_path)`")
    except Exception as e:
        raise RuntimeError(f"Failed to load the first frame from `cv2.VideoCapture(video_path)` with error: `{e}`")

    # Video info
    info = {}
    info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["video_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["video_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
    info["duration_sec"] = info["frame_count"] / info["fps"]
    info["save_time"] = datetime.fromtimestamp(getmtime(video_path)).strftime('%d/%m/%Y %H:%M:%S')
    info["cv2_is_unstable"] = False
    info["as_string"] = f'saved: {info["save_time"]}  ' \
                        f'duration: {info["duration_sec"]} sec  ' \
                        f'frames: {info["frame_count"]}  ' \
                        f'WH: {(info["video_width"], info["video_height"])}  ' \
                        f'fps: {info["fps"]}'

    # ---------------------------------------
    # Simple attempt to determine frame count
    # ---------------------------------------

    # If cv2's own frame count is correct there's no need to do anything fancy
    if is_valid_end_frame(cap, info["frame_count"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, info
    else:
        msg = f"cv2's frame count is unreliable, this may be an indication that cv2 is struggling to read `{video_path=}`."
        warnings.warn(msg)
        info["cv2_is_unstable"] = True

    # ---------------------------------------
    # Binary search to determine frame count
    # ---------------------------------------

    # Find some reasonable starting values
    start_value = 0
    for end_value in [1, 100, 1000, 5000, 10_000, 100_000, int(1e6), int(1e7), int(1e8)]:
        failed_to_read = not frame_can_be_read(cap, end_value)
        if failed_to_read:
            break
        start_value = end_value

    # Binary search
    counter = 0
    while True and (counter <= max_iteration):
        counter += 1
        seperation_point = start_value + ((end_value - start_value) // 2)
        if not frame_can_be_read(cap, seperation_point):
            end_value = seperation_point
        else:
            start_value = seperation_point
        if (end_value - start_value) == 1:
            assert is_valid_end_frame(cap, start_value), "Something went wrong"
            info["frame_count"] = start_value
            info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
            info["as_string"] = f'saved: {info["save_time"]}  ' \
                                f'duration: {info["duration_sec"]} sec  ' \
                                f'frames: {info["frame_count"]}  ' \
                                f'WH: {(info["video_width"], info["video_height"])}  ' \
                                f'fps: {info["fps"]}'
            break
    if (counter > max_iteration) and (not is_valid_end_frame(cap, start_value)):
        raise RuntimeError(f"Failed to find the total number of frames after `{max_iteration}` iterations")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, info


def extract_equally_spaced_numbers(indexes, n):
    if n < 2:
        raise ValueError(f"`{n=}` should be greater than or equal to 2.")
    if len(indexes) < n or len(indexes) < 2:
        raise ValueError(f"Expected: `{n=}` <= `{len(indexes)=}` > 2")

    first_frame = indexes[0]
    last_frame = indexes[-1]

    # We subtract 2 from n to account for the inclusion of the first and last elements.
    indices = np.linspace(0, len(indexes) - 1, num=n, endpoint=True)
    extracted_indexes = [indexes[int(i)] for i in indices]

    # Ensure that the first and last frames are included.
    extracted_indexes[0] = first_frame
    extracted_indexes[-1] = last_frame
    assert len(extracted_indexes) == n, "Should not be possible"
    return extracted_indexes

__all__ = ["get_cv2_video_capture"]