from __future__ import annotations
import numpy as np
import PIL
from PIL import Image
import os
import requests
import cv2
import copy

# Mine
from __config import *
import __checker


def resize_and_bgr_parser(image, resize_factor, BGR2RGB):
    __checker.assert_resize_factor_is_ok(resize_factor)
    if resize_factor != 1.0:
        height = int(image.shape[0] * resize_factor)
        if height < global_config.image_min_height:
            raise ValueError(f"After resize the image's `{height=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                             f"You can change this threshold through: `global_config.min_height=<NEW_VALUE>`")
        width = int(image.shape[1] * resize_factor)
        if width < global_config.image_min_width:
            raise ValueError(f"After resize the image's `{width=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                             f"You can change this threshold through: `global_config.min_width=<NEW_VALUE>`")
        interpolation = cv2.INTER_AREA if (resize_factor < 1.0) else global_config.resize_interpolation
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    if BGR2RGB:
        image = image[:, :, ::-1]
    return image


# noinspection PyTypeChecker
def _parse_url(image_url:str) -> np.ndarray:
    __checker.assert_type(image_url, str, "image_url")
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"`{image_url=}` was interpreted as an image URL, but the get-request failed with status code: `{response.status_code}`")
    try:
        image = np.array(Image.open(response.raw))
    except Exception as e:
        raise RuntimeError(f"Failed to convert downloaded image to an np.array with error:\n{e}")

    try:
        return _parse_numpy(image)
    except Exception as e:
        raise RuntimeError(f"`{image_url=}` was interpreted as an image URL and successfully downloaded, "
                           f"but it could not be converted to standard format (uint8, RGB np.ndarray, range:0-255) due to the error: `{e}`")


def _parse_path(image_path) -> np.ndarray:
    try:
        image = cv2.imread(image_path)[:, :, ::-1] # [:, :, ::-1] is BGR->RGB
    except Exception as e:
        raise RuntimeError(f"`{image_path=}` was interpreted as an image path, but it could not be loaded due to the error: `{e}`")

    # TODO: Is this neccesary? cv2.imread always return a correctly formatted RGB image as far as I can tell
    try:
        return _parse_numpy(image)
    except Exception as e:
        raise RuntimeError(f"`{image_path=}` was interpreted as an image path and successfully loaded, "
                           f"but it could not be converted to standard format (uint8, RGB np.ndarray, range:0-255) due to the error: `{e}`")

def parse_torch_image_batch_as_uint8_rgb_numpy_array(
        torch_batch:torch.Tensor,
        resize_factor: float = 1.0,
        BGR2RGB:bool = False,
        attempt_inverse_standardization_if_necessary:bool=True
) -> List[ndarray]:
    """
    Convert an RGB `torch_batch` (N, 3, H, W) into a list of numpy images (H, W, 3).

    :param torch_batch: The input batch of images in torch.Tensor format (N, 3, H, W). Can handle (N, H, W, 3) as well.
    :param resize_factor: Resize factor for each image. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB.
    :param attempt_inverse_standardization_if_necessary: Flag to attempt inverse standardization if necessary (ImageNet std and mean are used).
           For customization use `global_config.torch_standardization_mean` and `global_config.torch_standardization_std`.
    :return: list of numpy images (H, W, 3) in a standardized format (uint8 with pixels value between 0-255).
    """

    # Checks
    if not TORCH_FOUND:
        raise RuntimeError("You are attempting to parse a torch image, but failed to find the torch library")
    __checker.assert_types(
        [torch_batch, resize_factor, BGR2RGB, attempt_inverse_standardization_if_necessary],
        [torch.Tensor, float, bool, bool],
        ["torch_batch", "resize_factor", "BGR2RGB", "attempt_inverse_standardization_if_necessary"]
    )
    __checker.assert_resize_factor_is_ok(resize_factor)

    # Ensure CPU
    torch_batch = torch_batch.detach().cpu()
    if "int" in str(torch_batch.dtype):
        torch_batch = torch_batch.int()
    elif "float" in str(torch_batch.dtype):
        torch_batch = torch_batch.float()
    else:
        raise ValueError(f"Expected `{torch_batch.dtype=}` to be some variant of a float or int")

    # Dimensions
    dims = len(torch_batch.shape)
    if dims != 4:
        raise ValueError(f"Expected `torch_batch` to be of shape (batch_size, channels, height, width), but found `{torch_batch.shape=}")

    # Color channels
    batch_size, c, h, w = torch_batch.shape
    if (c >= w) or (c >= h):
        if (w < global_config.image_min_width) and (h < global_config.image_min_height) and (c == 1 or c == 3):
            msg = f"Expected `{torch_batch.shape=}` to be of format (batch_size, channel, height, width),\n" \
                  f"but found something that will be interpreted as (batch_size, height, width, channel).\n" \
                  f"`torch_batch` will be permuted to fit this assumption which will yield: `{torch_batch.permute(0,3,1,2).shape=}`"
            warnings.warn(msg)
            torch_batch = torch_batch.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected `torch_batch` to be of shape (batch_size, channels, height, width), but found `{torch_batch.shape=}")

    if torch_batch.shape[1] not in [1, 3]:
        raise ValueError(f"Expected `torch_batch` to be of shape (batch_size, channels, height, width), where channels in [1,3], but found `{torch_batch.shape=}")

    # Batch size
    if not (batch_size > 0):
        raise ValueError(f"Expected `torch_batch` to be of shape (batch_size, channels, height, width), where batch_size>0, but found `{torch_batch.shape=}")

    # Width and height
    if torch_batch.shape[2] < global_config.image_min_height:
        raise ValueError(f"The height of the image `{torch_batch.shape[2]=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                         f"You can change this threshold through: `global_config.min_height=<NEW_VALUE>`")
    if torch_batch.shape[3] < global_config.image_min_width:
        raise ValueError(f"The width of the image `{torch_batch.shape[3]=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                         f"You can change this threshold through: `global_config.min_width=<NEW_VALUE>`")

    # Attempt converting all images in the batch
    images = []
    for i, image in enumerate(torch_batch):
        try:
            images.append(resize_and_bgr_parser(
                _parse_torch(image,attempt_inverse_standardization_if_necessary), resize_factor, BGR2RGB)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse the {i}/{batch_size} image in the batch due to error:\n{e}")
    return images

def parse_numpy_image_batch_as_uint8_rgb_numpy_array(numpy_batch:ndarray, resize_factor: float = 1.0, BGR2RGB:bool = False) -> List[ndarray]:
    """
    Convert an RGB `numpy_batch` (N, H, W, 3) into a list of numpy images (H, W, 3).

    :param numpy_batch: The input batch of images in torch.Tensor format (N, 3, H, W).
    :param resize_factor: Resize factor for each image. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB.
    :return: list of numpy images (H, W, 3) in a standardized format (uint8 with pixels value between 0-255).
    """

    # Checks
    __checker.assert_types([numpy_batch, resize_factor, BGR2RGB], [ndarray, float, bool], ["numpy_batch", "resize_factor", "BGR2RGB"])
    if not (len(numpy_batch.shape) == 4):
        raise ValueError(f"Expected `numpy_batch` to be of shape (batch_size, height, width, 3), but found `{numpy_batch.shape=}")
    batch_size, height, width, channels = numpy_batch.shape
    if batch_size == 0:
        raise ValueError(f"Expected `numpy_batch` to be of shape (batch_size, height, width, 3) with a batch_size>0, but found `{numpy_batch.shape=}")
    if batch_size > 1_000:
        msg = f"`{batch_size=}` is suspiciously high."
        warnings.warn(msg)
    if not (channels == 3):
        raise ValueError(f"Expected `numpy_batch` to be a color image of shape (batch_size, height, width, 3), but found `{numpy_batch.shape=}")
    if height < global_config.image_min_height:
        raise ValueError(f"The height of the image `{numpy_batch.shape[1]=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                         f"You can change this threshold through: `global_config.min_height=<NEW_VALUE>`")
    if width < global_config.image_min_width:
        raise ValueError(f"The width of the image `{numpy_batch.shape[2]=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                         f"You can change this threshold through: `global_config.min_width=<NEW_VALUE>`")
    __checker.assert_resize_factor_is_ok(resize_factor)

    # Attempt converting all images in the batch
    images = []
    for i, image in enumerate(numpy_batch):
        try:
            images.append(resize_and_bgr_parser(_parse_numpy(image), resize_factor, BGR2RGB))
        except Exception as e:
            raise RuntimeError(f"Failed to parse the {i}/{batch_size} image in the batch due to error:\n{e}")
    return images

def _parse_torch(torch_image:torch.Tensor, attempt_inverse_standardization_if_necessary:bool=True) -> np.ndarray:
    # ---------------------------------------------------------
    # Check and simple conversions
    # ---------------------------------------------------------
    if not TORCH_FOUND:
        raise RuntimeError("You are attempting to parse a torch image, but failed to find the torch library")
    __checker.assert_type(torch_image, torch.Tensor, "torch_image")

    # Ensure CPU
    torch_image = torch_image.detach().cpu().clone()
    if "int" in str(torch_image.dtype):
        torch_image = torch_image.int()
    elif "float" in str(torch_image.dtype):
        torch_image = torch_image.float()
    else:
        raise ValueError(f"Expected `{torch_image.dtype=}` to be some variant of a float or int")

    # Dimensions
    dims = len(torch_image.shape)
    if dims not in [2,3]:
        if (dims == 4) and (torch_image.shape[0] == 1):
            msg = f"Expected `{torch_image.shape=}` to be of format (channel, height, width),\n" \
                  f"but found something that will be interpreted as (1, channel, height, width).\n" \
                  f"`torch.image` will be squeezed to fit this assumption which will yield: `{torch_image.squeeze(0).shape=}`"
            warnings.warn(msg)
            torch_image = torch_image.squeeze(0)
            dims = len(torch_image.shape)
            assert dims == 3, "Should not be possible."
        else:
            extra = f"If `torch_image` is a batch (or similar) represented as a tensor, pass each instance individually." if (dims==4) else ""
            raise ValueError(f"Expected `torch_image` to be of shape (channels, height, width) or (height, width), but found `{torch_image.shape=}`.\n{extra}")

    # Color channels
    image_height, image_width = torch_image.shape[-2], torch_image.shape[-1]
    color_channels = None if (dims == 2) else torch_image.shape[0]
    if (dims == 3) and (color_channels >= image_width or color_channels >= image_height):
        h,w,c = torch_image.shape
        if (w < global_config.image_min_width) and (h < global_config.image_min_height) and (c == 1 or c==3):
            msg = f"Expected `{torch_image.shape=}` to be of format (channel, height, width) or (height, width),\n" \
                  f"but found something that will be interpreted as (height, width, channels).\n" \
                  f"`torch.image` will be permuted to fit this assumption which will yield: `{torch_image.permute(1,2,0).shape=}`"
            warnings.warn(msg)
            torch_image = torch_image.permute(1, 2, 0)
        else:
            raise ValueError(f"Expected `torch_image` to be of shape (channels, height, width) or (height, width), but found `{torch_image.shape=}`.")

    # Width and height
    if torch_image.shape[-1] < global_config.image_min_width:
        raise ValueError(f"The width of the image `{torch_image.shape[-1]=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                         f"You can change this threshold through: `global_config.min_width=<NEW_VALUE>`")
    if torch_image.shape[-2] < global_config.image_min_height:
        raise ValueError(f"The height of the image `{torch_image.shape[-2]=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                         f"You can change this threshold through: `global_config.min_height=<NEW_VALUE>`")

    # ---------------------------------------------------------
    # Standardized
    # ---------------------------------------------------------

    # Setup
    t_min, t_max, t_std, t_mean = torch.min(torch_image), torch.max(torch_image), torch.std(torch_image.float()), torch.mean(torch_image.float())
    is_not_normalized_image = (torch_image.min() < -0.01).any() and (torch_image.max() > 1.01).any()
    range_assumed_standardized = (-10 < t_min) and (t_max < 10)
    std_and_mean_assumed_standardized = (-1 < t_std) and (t_std < 3.0) and (-2 < t_mean) and (t_mean < 2)

    assumed_standardized = is_not_normalized_image and range_assumed_standardized and std_and_mean_assumed_standardized and global_config.try_inverse_torch_standardization
    assumed_rgb_image = (len(torch_image.shape) == 3) and (torch_image.shape[0] == 3)

    # If `torch_image` is assumed to have been standardized: Attempt to inverse the standardization
    if (assumed_standardized and assumed_rgb_image) and attempt_inverse_standardization_if_necessary:
        mean = torch.as_tensor(global_config.torch_standardization_mean, dtype=torch_image.dtype).view(-1, 1, 1)
        std = torch.as_tensor(global_config.torch_standardization_std, dtype=torch_image.dtype).view(-1, 1, 1)
        torch_image.mul_(std).add_(mean)
    if (assumed_standardized and (not assumed_rgb_image)) and attempt_inverse_standardization_if_necessary:
        msg = f"`torch_image` appears to have been standardized, but cannot attempt to inverse the standardization, " \
              f"Because {torch_image.shaoe=} does not match a typical RGB tensor image of shape (3, height, width)"
        warnings.warn(msg)

    # ---------------------------------------------------------
    # Attempt np.ndarray conversion
    # ---------------------------------------------------------

    # (c, h, w) --> (h, w, c) if necessary
    if len(torch_image.shape) == 3:
        torch_image = torch_image.permute(1, 2, 0)

    try:
        return _parse_numpy(torch_image.numpy())
    except Exception as e:
        raise RuntimeError(f"Failed to convert torch-image to np.array with error:\n{e}")


def _parse_numpy(numpy_image:np.ndarray) -> np.ndarray:
    # TODO: What happens if you pass some that's (h, w, 2)?? And what should happened?
    # ---------------------------------------------------------
    # Check
    # ---------------------------------------------------------
    __checker.assert_type(numpy_image, np.ndarray, "numpy_image")

    # Dimensions
    dims = len(numpy_image.shape)
    if dims not in [2,3]:
        extra = f"If `numpy_image` is a video (or similar) represented as an np.ndarray try passing each frame individually." if (dims==4) else ""
        raise ValueError(f"Expected `numpy_image` to be of shape (height, width, channels) or (height, width), but found `{numpy_image.shape=}`.\n{extra}")

    # Color channels
    image_height, image_width = numpy_image.shape[0], numpy_image.shape[1]
    color_channels = None if (dims == 2) else numpy_image.shape[2]
    if (dims == 3) and (color_channels >= image_width or color_channels >= image_height):
        raise ValueError(f"Expected `numpy_image` to be of shape (height, width, channels) or (height, width), but found `{numpy_image.shape=}`.\n"
                         f"If you have been dealing with tensors: `<YOUR_NUMPY_IMAGE>.transpose(1,2,0)` might fix the issue.")

    # Width and height
    if numpy_image.shape[1] < global_config.image_min_width:
        raise ValueError(f"The width of the image `{numpy_image.shape[1]=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                         f"You can change this threshold through: `global_config.min_width=<NEW_VALUE>`")
    if numpy_image.shape[0] < global_config.image_min_height:
        raise ValueError(f"The height of the image `{numpy_image.shape[0]=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                         f"You can change this threshold through: `global_config.min_height=<NEW_VALUE>`")

    # ---------------------------------------------------------
    # Color channels
    # ---------------------------------------------------------

    # setup
    has_alpha = (dims == 3) and numpy_image.shape[2] == 4
    is_2_channel_greyscale = dims == 2
    is_3_channel_greyscale = (dims == 3) and (numpy_image.shape[-1] == 1)

    # Remove alpha channel (for compatibility)
    if has_alpha:
        numpy_image = numpy_image[:, :, :3]

    # Adds 3 identical channels to greyscale images (for compatibility)
    elif is_3_channel_greyscale:
        numpy_image = np.repeat(numpy_image, 3, axis=2)
    elif is_2_channel_greyscale:
        numpy_image = np.repeat(np.expand_dims(numpy_image, 2), 3, axis=2)

    # ---------------------------------------------------------
    # Data type
    # ---------------------------------------------------------

    # Setup
    data_type = str(numpy_image.dtype)
    is_float = data_type.startswith("float")
    is_unsigned_int = data_type.startswith("uint")
    is_int = data_type.startswith("int")

    # If the dtype is not recognized
    if not any([is_float, is_unsigned_int, is_int]):
        msg = f"Expected `{str(numpy_image.dtype)=}` to be a uint, int or float. Will attempt auto conversion, but this may well fail!"
        warnings.warn(msg)

        # Attempt auto conversion for all 3 acceptable dtypes
        for dtype in ("uint", "int", "float"):
            try:
                numpy_image_new_type = copy.deepcopy(numpy_image).astype(dtype)
                numpy_image_new_type = _parse_numpy(numpy_image_new_type)
                return numpy_image_new_type
            except Exception as e:
                continue

        raise ValueError(f"Expected `{str(numpy_image.dtype)=}` to be a uint, int or float.")

    if is_float:
        if (0.0 <= numpy_image).all() and (numpy_image <= 1.0).all():
            numpy_image = (numpy_image * 255).astype(np.uint8)
        elif (0.0 <= numpy_image).all() and (numpy_image <= 255.0).all():
            numpy_image = numpy_image.astype(np.uint8)
        elif (global_config.image_normalization_clip_min <= numpy_image).all() and (numpy_image <= global_config.image_normalization_clip_max).all(): # TODO: Check
            msg = f"`numpy_image` is assumed to be a normalized (pixel values within 0-1) image, but one or more pixel values had to be clipped because they were outside 0-1." \
                  f"You can modify the minimum and maximum thresholds to control when clipping is allowed through: " \
                  f"`global_config.image_normalization_clip_min=<NEW_VALUE>` and `global_config.image_normalization_clip_max=<NEW_VALUE>`"
            warnings.warn(msg)
            numpy_image = (numpy_image.clip(0,1) * 255).astype(np.uint8)

    elif is_unsigned_int or is_int:
        # If black and white / binary image, cast to 0-255 range
        if np.all(np.logical_or(numpy_image == 0, numpy_image == 1)):
            numpy_image = (numpy_image * 255).astype(np.uint8)
        elif (0 <= numpy_image).all() and (numpy_image <= 255).all():
            numpy_image = numpy_image.astype(np.uint8)
        elif (global_config.image_clip_min <= numpy_image).all() and (numpy_image <= global_config.image_clip_max).all(): # TODO: Check
            msg = f"`numpy_image` is assumed to be a normal (pixel values within 0-255) image, but one or more pixel values had to be clipped because they were outside 0-255." \
                  f"You can modify the minimum and maximum thresholds to control when clipping is allowed through: " \
                  f"`global_config.image_clip_min=<NEW_VALUE>` and `global_config.image_clip_max=<NEW_VALUE>`"
            warnings.warn(msg)
            numpy_image = (numpy_image.clip(0, 255)).astype(np.uint8)
        # TODO: add checks for High Dynamic Range (HDR) 16/24 bits pictures

    return numpy_image


# noinspection PyTypeChecker
def _parse_pillow(image_pillow:PIL.Image.Image) -> np.ndarray:
    __checker.assert_type(image_pillow, PIL.Image.Image, "image_pillow")
    try:
        image_pillow = np.array(image_pillow)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PIL-image to np.array with error:\n{e}")
    return _parse_numpy(image_pillow)

def parse_image_as_uint8_rgb_numpy_array(image_source: ImageType, resize_factor: float = 1.0, BGR2RGB:bool = False) -> ndarray:
    """
    Convert `image_source` into a standardized numpy array - shape: (H, W, 3), pixels: uint8 with pixels value between 0-255.

    :param image_source: An image. Can handle most image sources: ndarray, url, path, torch, PIL. For a full description see: TODO insert link
    :param resize_factor: Resize factor for each image. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB.
    :return: a numpy image (H, W, 3) in a standardized format (uint8 with pixels value between 0-255).
    """

    # Type checks
    __checker.assert_image_type(image_source, "image_source")
    __checker.assert_types([resize_factor, BGR2RGB], [float, bool], ["resize_factor", "BGR2RGB"])
    __checker.assert_resize_factor_is_ok(resize_factor)

    # Determine what kind of data we're dealing with
    is_path = os.path.exists(image_source) if isinstance(image_source, str) else False
    is_url = True if (isinstance(image_source, str) and __checker.is_valid_url(image_source)) else False
    is_ndarray = True if isinstance(image_source, np.ndarray) else False
    is_torch_tensor = True if (TORCH_FOUND and isinstance(image_source, torch.Tensor)) else False
    is_pillow = True if isinstance(image_source, PIL.Image.Image) else False

    # Sanity checks
    if is_path and is_url:
        raise AssertionError("This should not be possible. Don't see how `image_source` can be a path and a url simultaneously.")
    if not any([is_path, is_url, is_ndarray, is_torch_tensor, is_pillow]):
        if isinstance(image_source, str) and (not os.path.exists(image_source)):
            raise ValueError(f"`{image_source=}` is a string, but failed to detect it as a valid path/URL. Did you perhaps provide a bad path/url?")
        raise AssertionError(f"This should not be possible. `image_source` of type `{type(image_source)=}` should have been accepted/rejected as a valid/invalid type by now.")

    # Distribute according to detected type
    if is_path:
        image = _parse_path(image_source)
    elif is_url:
        image = _parse_url(image_source)
    elif is_ndarray:
        image = _parse_numpy(image_source)
    elif is_torch_tensor:
        image = _parse_torch(image_source)
    elif is_pillow:
        image = _parse_pillow(image_source)
    else:
        raise AssertionError("This should not be possible.")

    # Remove potential alpha channels
    if image.shape[2] == 4:
        image = image[:, :, :3]

    shape_info = image_source.shape if hasattr(image_source, "shape") else None
    extra_debug_info = \
         f"\nHere's some debug info for the image: " \
         f"\n{'-'*25 + ' types ' + '-'*25}\n"\
         f"type (input)  =\t{type(image_source)}\n"\
         f"type (output) =\t{type(image.dtype)}" \
         f"\n\n{'-' * 25 + ' shape ' + '-' * 25}\n" \
         f"shape (input)  =\t{shape_info}\n" \
         f"shape (output)  =\t{image.shape}" \
         f"\n\n{'-' * 25 + ' pixel ' + '-' * 25}\n" \
         f"min=\t{np.min(image)}\n"\
         f"max=\t{np.max(image)}\n"\
         f"mean=\t{np.mean(image)}\n" \
         f"median=\t{np.median(image)}\n"\
         f"std=\t{np.std(image)}\n"


    # Final check
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Failed to convert `{type(image_source)=}` to an np.ndarray for unknown reasons.{extra_debug_info}")
    if (image < 0).any() and (image > 255).any():
        raise ValueError(f"The pixel values of `image_source` are not valid and all automatic conversions failed.{extra_debug_info}")
    if image.dtype != np.uint8:
        raise ValueError(f"Failed to convert `{image_source.dtype=}` to dtype uint8 for unknown reasons.{extra_debug_info}")
    if image.shape[1] < global_config.image_min_width:
        raise ValueError(f"The width of the image `{image.shape[1]=}` is below the minimum threshold `{global_config.image_min_width}`.\n"
                         f"You can change this threshold through: `global_config.image_min_width=<NEW_VALUE>`")
    if image.shape[0] < global_config.image_min_height:
        raise ValueError(f"The height of the image `{image.shape[0]=}` is below the minimum threshold`{global_config.image_min_height}`.\n"
                         f"You can change this threshold through: `global_config.image_min_height=<NEW_VALUE>`")
    if (len(image.shape) != 3) or (image.shape[2] != 3):
        raise ValueError(f"Failed to convert `image_source` to standard shape (height, width, 3): `{image.shape=}`.{extra_debug_info}")

    # Modifications
    if resize_factor != 1.0:
        width = int(image.shape[1] * resize_factor)
        height = int(image.shape[0] * resize_factor)
        interpolation = cv2.INTER_AREA if (resize_factor < 1.0) else global_config.resize_interpolation
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    if BGR2RGB:
        image = image[:, :, ::-1]

    return image


def threshold_image_count(image_source):
    warning_msg = f"Received `{len(image_source)}` images, the maximum limit is `{global_config.show_max_image_amount}`. " \
      f"Will pick `{global_config.show_max_image_amount}` random images from `image_source` for display and discard the rest.\n" \
      f"If you wish to change this limit, you can set `global_config.show_max_image_amount = <NEW LIMIT>`"
    warnings.warn(warning_msg)
    random_indexes = np.random.choice(np.arange(len(image_source)), global_config.show_max_image_amount, replace=False)
    images = [image_source[i] for i in sorted(random_indexes)]
    return images

def parse_arbitrary_image_source(image_source:ImageSource, resize_factor: float = 1., BGR2RGB: bool = False) -> List[ndarray]:
    """
    Convert image(s) from `image_source` into a standardized numpy format - shape: (H, W, 3), pixels: uint8 with pixels value between 0-255.

    :param image_source: A image or list of images. Can handle most image sources: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param resize_factor: Resize factor for each image. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB.
    :return: a list of numpy images (H, W, 3) in a standardized format (uint8 with pixels value between 0-255).
    """

    # Checks
    if not torch:
        __checker.assert_in(type(image_source), [np.ndarray, PIL.Image.Image, str, list, tuple], "image_source")
    else:
        __checker.assert_in(type(image_source), [np.ndarray, torch.Tensor, PIL.Image.Image, str, list, tuple], "image_source")
    __checker.assert_types([resize_factor, BGR2RGB], [float, bool], ["resize_factor", "BGR2RGB"])
    if isinstance(image_source, (list, tuple)):
        if len(image_source) == 0:
            raise ValueError("`image_source` should contain image-information, but it's empty. Did you perhaps pass in an empty list or something similar?")
        for i, item in enumerate(image_source):
            __checker.assert_image_type(item, f"image_source[{i}]")
    __checker.assert_resize_factor_is_ok(resize_factor)


    # Prepare `image_source` for parsing
    if not isinstance(image_source, (list, tuple)):
        image_source = [image_source]
    if len(image_source) > global_config.show_max_image_amount:
        image_source = threshold_image_count(image_source)

    # Parse `image_source`
    images = []
    for item in image_source:
        if (not isinstance(item, PIL.Image.Image)) and ((item is None) or (len(item) == 0)):
            raise ValueError(f"`image_source` itself or one of its items are invalid: `{item=}`")
        elif isinstance(item, torch.Tensor) and __checker.assumed_torch_batch(item):
            images += parse_torch_image_batch_as_uint8_rgb_numpy_array(item, resize_factor, BGR2RGB)
        elif isinstance(item, np.ndarray) and __checker.assumed_numpy_batch(item):
            try:
                images += parse_numpy_image_batch_as_uint8_rgb_numpy_array(item, resize_factor, BGR2RGB)
            except Exception as e:
                raise ValueError(f"Received a numpy array of shape {item.shape=} which was interpreted as a batch of images (N, height, width, channel)."
                                 f"But failed to parse it as such. Error:\n{e}")
        elif isinstance(item, str) and any(item.lower().endswith(f) for f in global_config.video_formats_all):
            import __video
            images.append(
                __video.show_video(
                    item, verbose=global_config.in_debug_mode, display_image=False, return_image=True,
                    max_output_image_size_wh=None, resize_factor=resize_factor, BGR2RGB=BGR2RGB
                )
            )
        else:
            images.append(parse_image_as_uint8_rgb_numpy_array(item, resize_factor, BGR2RGB))
        if len(images) > global_config.show_max_image_amount:
            images = threshold_image_count(images)
            break

    assert len(images) > 0, "This should not be possible"
    return images

def quick_parse_image_source(image_source:ImageSource) -> list[ndarray]:
    if not isinstance(image_source, (tuple, list)):
        image_source = [image_source]
    if len(image_source) > global_config.show_max_image_amount:
        image_source = threshold_image_count(image_source)
    for item in image_source:
        __checker.quick_assert_valid_image(item)
    return image_source


__all__ = [
    "parse_torch_image_batch_as_uint8_rgb_numpy_array",
    "parse_numpy_image_batch_as_uint8_rgb_numpy_array",
    "parse_image_as_uint8_rgb_numpy_array",
    "parse_arbitrary_image_source"
]
