from __future__ import annotations
from PIL import Image
import cv2

from __config import *
import __utils
import __checker
import __image_modifier
import __grid
import __collage
import __parser


def show_image(
    image_source:ImageType,
    title:Optional[str]=None,
    display_image: bool = True,
    return_image: bool = False,
    max_output_image_size_wh: tuple = None,
    save_image_path:Optional[str] = None,
    resize_factor: float = 1.0,
    BGR2RGB: bool = False,
    parse_image:bool=True
) -> Optional[ndarray]:
    """
    Display a single image from `image_source`, regardless of its format.

    :param image_source: An image. Can handle most image sources: ndarray, url, path, torch, PIL. For a full description see: TODO insert link
    :param title: Custom title for the displayed frames. If not provided, the video file name will be used as the title.
    :param display_image: If True, display the processed image using cv2.imshow() or PIL if in jupyter.
    :param return_image: If True, return the processed image as a np.ndarray (H, W, 3).
    :param max_output_image_size_wh: Maximum width and height of the displayed/returned image. If None, nothing will be done.
    :param save_image_path: If provided, save the processed image at the given file path. Default is None (no saving).
    :param resize_factor: Resize factor for each image/frame. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB before display/return.
    :param parse_image: If True, will thoroughly check image_source and ensure it's in the correct format

    """

    # Init
    __checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image)
    if parse_image:
        image_source = __parser.parse_image_as_uint8_rgb_numpy_array(image_source, resize_factor, BGR2RGB)
    else:
        __checker.quick_assert_valid_image(image_source)

    # Universal stuff
    if title is not None:
        image_source = __image_modifier.draw_image_title(image_source, title)
    if max_output_image_size_wh is not None:
        image_source = __image_modifier.resize_universal_output_image(image_source, max_output_image_size_wh=max_output_image_size_wh)
    if save_image_path is not None:
        __utils.save_image_to_disk(save_image_path, image_source)

    if display_image:
        # Display in jupyter notebook
        if IN_JUPYTER:
            try:
                display(Image.fromarray(image_source))
            except Exception as e:
                warning_message = f"Failed to display `image` in jupyter due to the following error: `{e}`."
                warnings.warn(warning_message)

        # Display with cv2
        else:
            try:
                cv2.imshow("Just show it", image_source[:, :, ::-1]) # [:, :, ::-1] converts RGB -> BGR
                cv2.waitKey(global_config.cv2_wait_key_delay_ms)
                cv2.destroyAllWindows()
            except Exception as e:
                warning_message = f"Failed to display `image` with cv2 due to the following error: `{e}`."
                warnings.warn(warning_message)
                cv2.destroyAllWindows()

    if return_image:
        return image_source


def show(
    image_source:ImageSource,
    title: Optional[str] = None,
    display_image: bool = True,
    return_image: bool = False,
    max_output_image_size_wh: Optional[tuple] = (1920, 1080),
    save_image_path: Optional[str] = None,
    resize_factor: float = 1.0,
    BGR2RGB: bool = False,
    parse_image: bool = True,
):

    """
    Display `image_source` in an appropriate way, regardless of its format.

    :param image_source: An image or list of images. Can handle most image sources: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param title: Custom title for the displayed frames. If not provided, the video file name will be used as the title.
    :param display_image: If True, display the processed image using cv2.imshow() or PIL if in jupyter.
    :param return_image: If True, return the processed image as a np.ndarray (H, W, 3).
    :param max_output_image_size_wh: Maximum width and height of the displayed/returned image. If None, nothing will be done.
    :param save_image_path: If provided, save the processed image at the given file path. Default is None (no saving).
    :param resize_factor: Resize factor for each image/frame. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB before display/return.
    :param parse_image: If True, will thoroughly check image_source and ensure it's in the correct format
    :returns: If return_image is True, return an image of shape np.ndarray (H, W, 3), otherwise None
    """

    # Checks
    # --------------------------------------------------------------------
    # # Universal stuff
    # --------------------------------------------------------------------

    __checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image)
    if parse_image:
        image_source = __parser.parse_arbitrary_image_source(image_source, resize_factor, BGR2RGB)
    else:
        image_source = __parser.quick_parse_image_source(image_source)

    # Find out how best to display the images extracted from `source`
    if isinstance(image_source, list) and (len(image_source) == 1):
        final_image = image_source[0]
    elif max(__grid.get_all_possible_aspect_ratios(image_source)) < 0.1:
        final_image = __grid.show_grid(image_source, display_image=False, return_image=True, max_output_image_size_wh=(3000, 3000))
    else:
        final_image = __collage.show_collage(image_source, display_image=False, return_image=True, max_output_image_size_wh=(3000, 3000))

    # Wrap up
    if title:
        final_image = __image_modifier.draw_image_title(final_image, title)
    if max_output_image_size_wh is not None:
        final_image = __image_modifier.resize_universal_output_image(final_image, max_output_image_size_wh=max_output_image_size_wh)
    if save_image_path is not None:
        __utils.save_image_to_disk(save_image_path, final_image)
    if display_image:
        show_image(final_image)
    if return_image:
        return final_image


__all__ = ["show_image", "show"]