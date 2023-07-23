from __future__ import annotations
import cv2
import numpy as np
import copy
from PIL import Image, ImageDraw

from __config import *
import __checker
import __parser
import __utils

def resize_image_respect_aspect_ratio(image, width, height, interpolation=cv2.INTER_CUBIC):
    if (width is not None) and (height is None):
        ratio = width / image.shape[1]
        height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    elif (height is not None) and (width is None):
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    elif (height is not None) and (width is not None):
        image, *_ = resize_image_respect_aspect_ratio(image, width=width, height=None)
        image, *_ = resize_image_respect_aspect_ratio(image, width=None, height=height)

    h, w, _ = image.shape
    return image, h, w


def resize_universal_output_image(output_image:np.ndarray, max_output_image_size_wh:tuple):
    __checker.assert_types([output_image, max_output_image_size_wh],
                           [np.ndarray, tuple], ["output_image", "max_output_image_size_wh"], [0, 1])
    if max_output_image_size_wh is None:
        return output_image

    # Resize output image if user wants it
    max_w_out, max_h_out = max_output_image_size_wh
    final_h, final_w, _ = output_image.shape

    if max_w_out and (final_w < max_w_out):
        max_w_out = None
    if max_h_out and (final_h < max_h_out):
        max_h_out = None

    if max_w_out and max_h_out:
        output_image, *_ = resize_image_respect_aspect_ratio(output_image, width=max_w_out, height=max_h_out)
    elif max_w_out or max_h_out:
        output_image, *_ = resize_image_respect_aspect_ratio(output_image, width=max_w_out, height=max_h_out)

    return output_image


def center_pad_images_into_nparray(source):
    max_height = max(img.shape[0] for img in source)
    max_width = max(img.shape[1] for img in source)

    padded_images = []
    for img in source:
        height, width, _ = img.shape
        pad_height = (max_height - height) // 2
        pad_width = (max_width - width) // 2

        pad_top = pad_height
        pad_bottom = max_height - (height + pad_height)
        pad_left = pad_width
        pad_right = max_width - (width + pad_width)

        padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        padded_images.append(padded_img)

    return np.array(padded_images)


def draw_text_cv2(
        image_source: ImageType,
        text: str,
        position: Union[str, tuple] = "upper_left_corner",
        font: int = cv2.FONT_HERSHEY_DUPLEX,
        font_scale: int = None,
        color: tuple = None,
        thickness=None,
        background_color: Union[str, tuple] = "auto",
        line_type: int = cv2.LINE_AA,
        padding_pixels: int = 10,
        deep_copy:bool=True
):
    """
    A function designed to draw `text` on `image` with minimum effort.

    # EXAMPLE
    >> image = np.zeros((512, 1024, 3)).astype(np.uint8)
    >> draw_text(image, "Hello 123", "bottom_left_corner", padding_pixels=100)

    :param image_source: An image. Can handle most image sources: ndarray, url, path, torch, PIL. For a full description see: TODO insert link
    :param text: The text string to be drawn on the image.
    :param position: The position of the text on the image. It can be specified as a string:
                     ("upper_left_corner", "upper_right_corner", "bottom_left_corner", "bottom_right_corner", or "center")
                     or as a tuple of (x, y) coordinates.
    :param font: The font type to be used for the text. It should be one of the predefined constants from the cv2 library (e.g., cv2.FONT_HERSHEY_DUPLEX).
    :param font_scale: The scale factor for the font size. If not provided, it will be calculated based on the image size.
    :param color: The color of the text, specified as a tuple of RGB values. If not provided, it will be automatically determined based on the background color of `image`.
    :param thickness: The thickness of the text. If not provided, it will be calculated based on the image size.
    :param background_color: The background color behind the text. It can be specified as a string ("auto") or as a tuple of RGB values. If "auto" is specified,
                             the background color will be automatically determined based on the background color of `image`.
                             If a tuple is provided, the background color will be filled with that color.
    :param line_type: The type of line used for drawing the text. It should be one of the predefined constants from the cv2 library: e.g `cv2.LINE_AA=16`
    :param padding_pixels: The number of pixels used for padding around the text. This can only be applied together with string `position` e.g. "upper_left_corner"
    :param deep_copy: If true, copy `image` before drawing text on it
    """

    # Checks - simple
    # ------------------------------------------------------
    __checker.assert_types(
        [text, position, font, font_scale, color, thickness, background_color, line_type, padding_pixels],
        [str, (str, tuple), int, int, tuple, int, (str, tuple), int, int],
        ["text", "position", "font", "font_scale", "color", "thickness", "background_color", "line_type", "padding_pixels"],
        [0, 1, 0, 1, 1, 1, 1, 0, 0],
    )

    # Checks - image
    image_source = __parser.parse_image_as_uint8_rgb_numpy_array(image_source)
    __checker.assert_valid_image(image_source)
    if deep_copy:
        image_source = copy.deepcopy(image_source)
    if (len(image_source.shape) == 3) and (image_source.shape[-1] == 1):
        warnings.warn("You have provided an greyscale image in the format (dim1 x dim2 x 1), it will be squeezed down to (dim1 x dim2)")
        image_source = image_source[:, :, 0:1].squeeze(2)

    # Checks - cv2 specific stuff and background_color
    # ------------------------------------------------------
    if font not in [0, 1, 2, 3, 4, 5, 6, 7]:
        raise ValueError(
            f"Unrecognized `{font=}`. As of writing this, there's 8 valid fonts in cv2. These are encoded with an enum spanning from 0-7 (both ends included).")
    if line_type not in [4, 8, 16]:
        raise ValueError(
            f"Unrecognized `{line_type=}`. As of writing this, there's 3 valid fonts in cv2. These are encoded with an enum with [LINE_4=4, LINE_8=8, LINE_AA=16]")
    if font_scale is not None:
        assert font_scale > 0, f"Expected font_scale > 0, but received `{font_scale=}`"
    if color is not None:
        assert (len(color) == 3) and all(0 <= c <= 255 for c in color), f"Expected `color` to contain 3 RGB values between 0-255, but received `{color=}`"
    if thickness is not None:
        assert thickness > 0, f"Expected thickness > 0, but received `{thickness=}`"
    if isinstance(background_color, tuple):
        assert (len(background_color) == 3) and all(0 <= c <= 255 for c in background_color), \
            f"Expected `background_color` to contain 3 RGB values between 0-255, but received `{background_color=}`"
    if isinstance(background_color, str):
        assert background_color == "auto", f"The only valid string-argument for background_color is 'auto', but received {background_color=}"

    # Checks - Text position
    # ------------------------------------------------------
    legal_positions = ["upper_left_corner", "upper_right_corner", "bottom_left_corner", "bottom_right_corner", "center"]
    if isinstance(position, str):
        assert position in legal_positions, f"Invalid `{position=}`. Legal values are: `{legal_positions}`"
    assert 0 <= padding_pixels <= min([c // 2 for c in image_source.shape[:2]]), \
        f"Expected 0 <= padding_pixels <= min_image_size/2, but received `{padding_pixels=}`"

    if isinstance(position, tuple):
        assert len(
            position) == 2, f"Invalid `{position=}`. Expected a tuple with (text_left_corner_x, text_left_corner_right)"
        assert 0 <= position[0] <= image_source.shape[
            0], f"Invalid `{position[0]=}`. The first coordinate (width) is outside the image"
        assert 0 <= position[1] <= image_source.shape[
            1], f"Invalid `{position[1]=}`. The second coordinate (height) is outside the image"

        if padding_pixels:
            warning_message = f"`{padding_pixels=}` will be ignored. It only works with string `position` e.g. `upper_left_corner`"
            warnings.warn(warning_message)

    # Assign font_scale and thickness dynamically
    # ------------------------------------------------------
    if font_scale is None:
        font_scale = max(1, round(max(image_source.shape[:2]) / 1000))
    if thickness is None:
        thickness = max(1, round(max(image_source.shape[:2]) / 1000))

    # Handle position conversion
    # ------------------------------------------------------
    image_height, image_width = image_source.shape[0], image_source.shape[1]
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if isinstance(position, str):
        if position == "upper_left_corner":
            position = (padding_pixels, text_height + padding_pixels)
        elif position == "upper_right_corner":
            position = (image_width - (text_width + padding_pixels), text_height + padding_pixels)
        elif position == "bottom_left_corner":
            position = (padding_pixels, image_height - padding_pixels)
        elif position == "bottom_right_corner":
            position = (image_width - (text_width + padding_pixels), image_height - padding_pixels)
        elif position == "center":
            position = ((image_width - text_width) // 2, (image_height + text_height) // 2)
        else:
            raise RuntimeError("Shouldn't have gotten this far")

    # Handle text and background colors
    # ------------------------------------------------------

    # Calculate some shared parameters for color calculations
    if (color is None) or (background_color is not None):
        extra_padding = text_height // 8

        # Find text area in the image
        x1 = max(0, position[0] - extra_padding)
        x2 = min(image_width, position[0] + extra_padding + text_width)
        y1 = max(0, position[1] - extra_padding - text_height)
        y2 = min(image_height, position[1] + extra_padding)

        # Determine if the text should be dark or bright
        background_pixels_mean = image_source[y1:y2, x1:x2].mean()
        is_dark = (background_pixels_mean < 128)

    # Background color user defined
    if isinstance(background_color, tuple):
        # why `ascontiguousarray` -> see explanation at the bottom of function
        image_source = cv2.rectangle(np.ascontiguousarray(image_source, dtype=np.uint8), (x1, y1), (x2, y2), background_color, -1)
        background_pixels_mean = image_source[y1:y2, x1:x2].mean()  # Must be recalculated due to the drawn box
        is_dark = (background_pixels_mean < 128)

    # Text color
    if color is None:
        color = (200, 200, 200) if is_dark else (50, 50, 50)

    # Background color -auto
    if background_color == "auto":
        background_image = np.zeros(image_source[y1:y2, x1:x2].shape)
        if not is_dark:
            background_image = background_image + 255
        background_image = background_image.astype(np.uint8)
        image_source[y1:y2, x1:x2] = cv2.addWeighted(image_source[y1:y2, x1:x2], 0.40, background_image, 0.60, 0.0)

    # Draw text to image
    # ------------------------------------------------------
    try:
        image_source = cv2.putText(image_source, text, position, font, font_scale, color, thickness, line_type)
    except Exception:
        # TODO: Find out if this is still an issue. I encountered a weird cv2 bug in which it refused
        # to acknowledge `image` as a valid np.array unless `np.ascontiguousarray` was applied. This fix
        # was mentioned in an official bug report on cv2's github page. But it should not be necessary.
        image_source = np.ascontiguousarray(image_source, dtype=np.uint8)
        image_source = cv2.putText(image_source, text, position, font, font_scale, color, thickness, line_type)

    return image_source

class ImageModifier:
    # TODO
    """
    Grayscale Conversion: Convert a color image to grayscale by averaging the RGB channels.
    Grayscale 2 RGB: Simply duplicate x 3
    Gamma Correction: Adjust the image's brightness by applying a power law transformation.
    Histogram Equalization: Enhance the contrast of an image by redistributing the intensity values using the image's histogram.
    Histogram Stretching: Expand the range of pixel intensities to enhance the contrast of an image.
    Image Rotation: Rotate the image by a specified angle.
    Image Flipping: Flip the image horizontally or vertically.
    Image Cropping: Crop a region of interest from the image.
    Image Resizing: Resize the image to a specified width and height.
    Image Padding: Add padding to the image by extending the border.
    Image Blurring: Apply blurring filters to the image to reduce noise or smooth out details.
    Image Sharpening: Enhance the image's details and edges.
    Image Thresholding: Convert a grayscale image to binary by applying a threshold value.
    Image Filtering: Apply various image filters like Gaussian blur, median filter, or custom filters.
    Image Edge Detection: Detect edges in the image using techniques like Canny edge detection or Sobel operator.
    Image Morphological Operations: Perform operations like dilation, erosion, opening, and closing on the image.
    Image Color Adjustment: Adjust the image's color balance, saturation, or hue.
    """


# noinspection PyTypeChecker
def draw_text_pillow(
        image_source:ImageType,
        x:int,
        y:int,
        text:str,
        font_size:int=20,
        font_thickness:int=4,
        color:Tuple[int, int, int]=(25,25,25),
        background_color:tuple=(240,240,240),
        italic:bool=False
) -> ndarray:
    """
    Draw `text` on `image_source` at the specified position (x, y) using PIL.

    :param image_source: An image. Can handle most image sources: ndarray, url, path, torch, PIL. For a full description see: TODO insert link
    :param x: The x-coordinate for the starting position of the text.
    :param y: The y-coordinate for the starting position of the text.
    :param text: The text to be drawn on the image.
    :param font_size: The font size of the text.
    :param font_thickness: The thickness of the text stroke.
    :param color: The color of the text in RGB format.
    :param background_color: The background color of the text box in RGB format.
    :param italic: Flag to make the text italic.

    :return: The modified image with the text drawn.
    """

    # Checks
    __checker.assert_types(
        [x,   y,   text, font_size, font_thickness, color,  background_color, italic],
        [int, int, str,  int,       int,            tuple,  tuple,            bool],
        ["x", "y", "text", "font_size", "font_thickness", "color", "background_color", "italic"],
        [0,   0,   0,    0,         0,              0,      1,                0]
    )

    image_source = __parser.parse_image_as_uint8_rgb_numpy_array(image_source)
    __checker.assert_positive_int(x, zero_allowed=True, max_value_allowed=image_source.shape[1], variable_name="x")
    __checker.assert_positive_int(y, zero_allowed=True, max_value_allowed=image_source.shape[0], variable_name="y")
    __checker.assert_color(color, color_type="RGB", variable_name="color")
    if background_color is not None:
        __checker.assert_color(background_color, color_type="RGB", variable_name="background_color")

    # Draw
    font = __utils.get_font(font_size, font_thickness, italic)
    image_source = Image.fromarray(image_source)
    draw = ImageDraw.Draw(image_source)
    if background_color:
        left, top, right, bottom = draw.textbbox((x,y ), text, font=font)
        draw.rectangle((left-4, top-2, right+4, bottom+2), fill=background_color)
    draw.text((x, y), text, color, font=font)
    return np.array(image_source)


# noinspection PyTypeChecker
def draw_image_title(
        image_source:ImageType,
        text:str,
        placement:str= "center",
        font_size:int=30,
        font_thickness:int=4,
        text_color:Tuple[int, int, int]=(25, 25, 25),
        text_background_color:tuple=None,
        top_padding:int=None,
        top_padding_color:Tuple[int, int, int]=(255, 255, 255),
        italic:bool=False
) -> ndarray:
    """
    Draw a title `text` on `image` using PIL.

    :param image_source: An image. Can handle most image sources: ndarray, url, path, torch, PIL. For a full description see: TODO insert link
    :param text: The title text to be drawn on the image.
    :param placement: The placement of the text ('center', 'left', 'right').
    :param font_size: The font size of the text.
    :param font_thickness: The thickness of the text stroke (1,2,3,4,5,6,7).
    :param text_color: The color of the text in RGB format.
    :param text_background_color: The background color of the text box in RGB format.
    :param top_padding: The additional padding at the top of the image (if required).
    :param top_padding_color: The color of the top padding.
    :param italic: Flag to make the text italic.
    """

    # Checks
    __checker.assert_types(
        [text, placement, font_size, font_thickness, top_padding, italic],
        [str, str, int, int, int, bool],
        ["text", "placement", "font_size", "font_thickness", "top_padding", "italic"],
        [0, 0, 0, 0, 1, 0]
    )

    image_source = __parser.parse_image_as_uint8_rgb_numpy_array(image_source)
    __checker.assert_color(text_color, color_type="RGB", variable_name="text_color")
    if text_background_color is not None:
        __checker.assert_color(text_background_color, color_type="RGB", variable_name="text_background_color")
    if top_padding_color is not None:
        __checker.assert_color(top_padding_color, color_type="RGB", variable_name="top_padding_color")
    __checker.assert_in(placement, ["center", "left", "right"], "placement")

    # Setup
    image_height, image_width, _ = image_source.shape
    font = __utils.get_font(font_size, font_thickness, italic)

    # Determine text box dimensions
    image_source = Image.fromarray(image_source)
    left, top, right, bottom = ImageDraw.Draw(image_source).textbbox((0, 0), text, font=font)
    text_box_height = bottom - top; assert text_box_height > 0
    text_box_width = right - left; assert text_box_width > 0

    # Padding
    if top_padding is None:
        top_padding = int(text_box_height * 2.5)
    image_source = cv2.copyMakeBorder(np.array(image_source), top_padding, 0, 0, 0, cv2.BORDER_CONSTANT, None, value=top_padding_color)

    # Determine text placement
    y = top_padding // 2 - text_box_height // 2 - top  # `top` is the font's built-in "white space" on top.
    if placement == "center":
        x = (image_width - text_box_width) // 2
    elif placement == "left":
        x = 10
    elif placement == "right":
        x = image_width - text_box_width - 10
    else:
        raise ValueError("Shouldn't have gotten this far!")

    assert y > 0
    if x < 0:
        msg = f"`{text=}` with `{placement=}` exceeds the image's border, and as a result, parts of the text will not be visible."
        warnings.warn(msg)

    # Draw title
    image_source = Image.fromarray(image_source)
    draw = ImageDraw.Draw(image_source)
    if text_background_color:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle((left - 4, top - 2, right + 4, bottom + 2), fill=text_background_color)
    draw.text((x, y), text, text_color, font=font)
    return np.array(image_source)


def add_square_dropdown_shadow(image, shadow_upper_left_corner_xy, shadow_lower_right_corner_xy, shadow_offset_x, shadow_offset_y, shadow_color, background_color, iterations, blur_kernel):
    x1, y1 = shadow_upper_left_corner_xy
    x2, y2 = shadow_lower_right_corner_xy
    image[y1+shadow_offset_y:y2+shadow_offset_y, x1+shadow_offset_x:x2+shadow_offset_x, :] = shadow_color
    for _ in range(iterations):
        image = cv2.GaussianBlur(image, blur_kernel, 0)
    image[y1:y2, x1:x2, :] = background_color
    return image


__all__ = ["draw_text_cv2", "draw_text_pillow", "draw_image_title"]