from __future__ import annotations
import re
import numpy as np
import PIL
import os

from __config import *


def assert_path(path:str, variable_name:str) -> None:
    assert_types([path, variable_name], [str, str], ["variable_name", "path"])
    if not os.path.exists(path):
        raise ValueError(f"Received bad path: `{variable_name}`=`{path}`")

def assert_image_type(image_source, variable_name:str) -> None:
    assert_type(variable_name, str, variable_name)
    if not TORCH_FOUND:
        assert_in(type(image_source), [np.ndarray, PIL.Image.Image, str], variable_name)
    else:
        assert_in(type(image_source), [np.ndarray, torch.Tensor, PIL.Image.Image, str], variable_name)

def assert_valid_image(image: ndarray) -> None:
    """ Try and ensure that `image` is a legitimate formatted numpy image """
    assert_type(image, ndarray, 'image')

    # Check shape
    assert len(image.shape) in [2, 3], f"Expected the shape of the image to be in [2,3], but received `{len(image.shape)=}`"
    is_greyscale = (len(image.shape) == 2) or (len(image.shape) == 3 and (image.shape[2] == 1))
    if len(image.shape) != 2:
        assert image.shape[-1] in [1, 3, 4], f"Expected to find a greyscale image, RGB or RGBA, but found: `{image.shape=}`"
    if (len(image.shape) != 3) and (not is_greyscale):
        raise ValueError(f"Expected 3 dimensional shape for non-greyscale images (dim1 x dim2 x channels). "
                         f"Received shape `{image.shape}`")

    # Check data type
    if image.dtype != 'uint8':
        extra = ""
        if 'f' in str(image.dtype):
            extra = " If `image` is in 0.0-1.0 float format, try casting it with: `image = (image * 255).astype(np.uint8)`"
        raise TypeError(f"Expected `image` to be of dtype `uint8`, but received `{image.dtype}`.{extra}")

    # Check dimensions
    if image.shape[0] < 3:
        raise ValueError(f"Expected an image with a height of at least 3 pixels, but received `{image.shape[0]=}`")
    if image.shape[1] < 3:
        raise ValueError(f"Expected an image with a width of at least 3 pixels, but received `{image.shape[1]=}`")

    # Check pixel range
    if np.min(image) < 0:
        raise ValueError("Detected pixel value below 0. Consider using `numpy.clip` to get rid of them")
    if np.min(image) > 255:
        raise ValueError("Detected pixel value above 255. Consider using `numpy.clip` to get rid of them")

def is_valid_url(url:str) -> bool:
    """
    Attempt to check if `url` is valid.
    This function is taken more or less directly from the wonderful validators library - https://github.com/kvesteri/validators.
    All credit goes to them.
    """
    assert_type(url, str, "url")
    ip_middle_octet = r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5]))"
    ip_last_octet = r"(?:\.(?:0|[1-9]\d?|1\d\d|2[0-4]\d|25[0-5]))"

    regex = re.compile(
        r"^"
        r"(?:(?:https?|ftp)://)"
        r"(?:[-a-z\u00a1-\uffff0-9._~%!$&'()*+,;=:]+"
        r"(?::[-a-z0-9._~%!$&'()*+,;=:]*)?@)?"
        r"(?:"
        r"(?P<private_ip>"
        r"(?:(?:10|127)" + ip_middle_octet + r"{2}" + ip_last_octet + r")|"
        r"(?:(?:169\.254|192\.168)" + ip_middle_octet + ip_last_octet + r")|"
        r"(?:172\.(?:1[6-9]|2\d|3[0-1])" + ip_middle_octet + ip_last_octet + r"))"
        r"|"
        r"(?P<private_host>"
        r"(?:localhost))"
        r"|"
        r"(?P<public_ip>"
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"" + ip_middle_octet + r"{2}"
        r"" + ip_last_octet + r")"
        r"|"
        r"\[("
        r"([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|"
        r"([0-9a-fA-F]{1,4}:){1,7}:|"
        r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|"
        r"([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"
        r"([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
        r"([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"
        r"([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|"
        r"[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"
        r":((:[0-9a-fA-F]{1,4}){1,7}|:)|"
        r"fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|"
        r"::(ffff(:0{1,4}){0,1}:){0,1}"
        r"((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}"
        r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|"
        r"([0-9a-fA-F]{1,4}:){1,4}:"
        r"((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}"
        r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
        r")\]|"
        r"(?:(?:(?:xn--)|[a-z\u00a1-\uffff\U00010000-\U0010ffff0-9]-?)*"
        r"[a-z\u00a1-\uffff\U00010000-\U0010ffff0-9]+)"
        r"(?:\.(?:(?:xn--)|[a-z\u00a1-\uffff\U00010000-\U0010ffff0-9]-?)*"
        r"[a-z\u00a1-\uffff\U00010000-\U0010ffff0-9]+)*"
        r"(?:\.(?:(?:xn--[a-z\u00a1-\uffff\U00010000-\U0010ffff0-9]{2,})|"
        r"[a-z\u00a1-\uffff\U00010000-\U0010ffff]{2,}))"
        r")"
        r"(?::\d{2,5})?"
        r"(?:/[-a-z\u00a1-\uffff\U00010000-\U0010ffff0-9._~%!$&'()*+,;=:@/]*)?"
        r"(?:\?\S*)?"
        r"(?:#\S*)?"
        r"$",
        re.UNICODE | re.IGNORECASE
    )

    pattern = re.compile(regex)
    result = pattern.match(url)
    return result is not None


def assert_type(to_check, expected_type, variable_name:str, allow_none:bool=False) -> None:
    """
    Check object against expected type

    :param to_check: Object for type check
    :param expected_type: Expected type of `to_check`
    :param variable_name: The name of `to_check` that will be displayed in error messages
    :param allow_none: Weather or not None is acceptable
    """

    if not isinstance(allow_none, bool):
        error_message = f"Expected `allow_None` to by of type bool, but received `{type(allow_none)=}`"
        raise TypeError(error_message)
    if not isinstance(variable_name, str):
        error_message = f"Expected `variable_name` to by of type str, but received `{type(variable_name)=}`"
        raise TypeError(error_message)
    if (to_check is None) and (expected_type is None): # TODO: Test this
        error_message = f"`None` is not a valid type. If you're trying to check if `type(to_check) == None` try setting `expected_type=type(None)` instead."
        raise TypeError(error_message)

    is_ok = isinstance(to_check, expected_type)
    if allow_none:
        is_ok = (to_check is None) or is_ok

    if not is_ok:
        error_message = f"Expected type `type({variable_name})={type(to_check)}` to be of type `{str(expected_type)}`"
        raise TypeError(error_message)


def assert_types(to_check:list, expected_types:list, variable_names:List[str], allow_nones:list[int|bool]=None) -> None:
    """
    Check list of values against expected types

    :param to_check: List of values for type check
    :param expected_types: Expected types of `to_check`
    :param variable_names: The name of `to_check` that will be displayed in error messages
    :param allow_nones: list of booleans or 0/1
    """

    # Checks
    assert_type(to_check, list, "to_check")
    assert_type(expected_types, list, "expected_types")
    assert_type(variable_names, list, "variable_names")
    assert_type(allow_nones, list, "allow_nones", allow_none=True)
    if len(to_check) != len(expected_types):
        error_message = "length mismatch between `{to_check_values}` and `{expected_types}`"
        raise ValueError(error_message)

    # If `allow_nones` is None all values are set to False.
    if allow_nones is None:
        allow_nones = [False]*len(to_check)
    else:
        if not (len(variable_names) == len(allow_nones) == len(to_check)):
            raise ValueError(f"Expected equal lengths, but found: `{len(to_check)=}`, `{len(allow_nones)=}` and `{len(variable_names)=}`")
        for i, element in enumerate(allow_nones):
            if element in [0, 1]:
                allow_nones[i] = bool(element == 1) # the `== 1` is just to allow for zeros as False and ones as True
            else:
                raise ValueError(f"`{allow_nones=}` may only contain [False, True, 0, 1]")

    # check if all elements are of the correct type
    for v, t, n, a in zip(to_check, expected_types, variable_names, allow_nones):
        assert_type(to_check=v, expected_type=t, variable_name=n, allow_none=a)


def assert_list_types(to_check:Union[list, tuple], expected_type, variable_name:str, allow_none:bool=False) -> None:
    assert_type(to_check, (list, tuple), variable_name)
    for i, item in enumerate(to_check):
        assert_type(item, expected_type, f"{variable_name}[{i}]", allow_none)

def assert_dict_types(dict_name:str, dict_to_check:dict, names:List[str], types:list, must_be_present:list=None, allow_nones:list=None, eval_expr:List[str]=None, assert_dict_lte:bool=True) -> None:
    # Input checks
    assert_types([dict_name, dict_to_check, names, types, must_be_present, allow_nones, eval_expr, assert_dict_lte],
                 [str, dict, list, list, list, list, list, bool],
                 ["dict_name", "dict_to_check", "names", "types", "must_be_present", "allow_nones", "eval_expr", "assert_dict_lte"],
                 [0,0,0,0,1,1,1,0])
    assert_list_types(names, str, "names")

    # Check optionals and create dummies if None (avoid issues with different scenarios below that way)
    if must_be_present is not None: assert_list_types(must_be_present, (bool, int), "must_be_present")
    else: must_be_present = [False] * len(names)

    if allow_nones is not None: assert_list_types(allow_nones, (bool, int), "allow_nones")
    else: allow_nones = [None] * len(names)

    if eval_expr is not None: assert_list_types(names, str, "names")
    else: eval_expr = [None] * len(names)

    # Check lengths
    if (allow_nones is not None) and (must_be_present is not None):
        assert len(names) == len(types) == len(allow_nones) == len(must_be_present) == len(eval_expr), \
            f"Expected `{len(names)=}`, `{len(types)=}`, `{len(must_be_present)=}`, `{len(allow_nones)=}` and `{len(eval_expr)=}` to be of the same length"
    assert len(dict_to_check) <= len(names), f"Expected `len(dict_to_check) <= len(names)` but found `{len(dict_to_check)} <= {len(names)}`"

    # Check dict
    for name, type_, must_be, allow_none, eval_expr_ in zip(names, types, must_be_present, allow_nones, eval_expr):
        try:
            value = dict_to_check[name]
            assert_type(value, type_, f"{dict_name}[{name}]", bool(allow_none))
            if eval_expr_ is not None:
                assert_eval(value, eval_expr_, allow_none=True, name_of_check=name)
        except KeyError:
            assert must_be==False, f"Expected to find key:`{name}` in `{dict_name}={dict_to_check}`"


def assert_eval(to_check, eval_str: str, allow_none:bool=False, name_of_check:str=None):
    assert_types([eval_str, allow_none, name_of_check], [str, bool, str], ["eval_str", "allow_none", "name_of_check"], [0, 0, 1])
    if (to_check is None) and (allow_none is True):
        return

    assert "x" in eval_str, f"`{eval_str=}` must contain at least 1 'x' e.g. '0<=x<=1e6'"

    evaluation_exp = None
    try:
        evaluation_exp = eval_str.replace("x", str(to_check))
        success = eval(evaluation_exp)
    except Exception as e:
        raise RuntimeError(f"`Failed to execute the evaluation: {evaluation_exp=}` with the following error code:\n{e}")

    if not name_of_check:
        assert success, f"Evaluation failed: `{evaluation_exp}`"
    else:
        assert success, f"Evaluation of `{evaluation_exp}` with `{name_of_check}={to_check}` failed"


def assert_in(to_check, check_in, variable_name:str):
    """
    Check if the value `to_check` is present in `check_in`
    :param to_check: Value to be checked
    :param check_in: Values `to_check` is being checked against
    :param variable_name: Name which will be display with the assert fail
    """
    try:
        is_in = to_check in check_in
    except Exception:
        raise RuntimeError(f"Failed to execute `{to_check} in {check_in}`")

    if not is_in:
        raise ValueError(f"Expected `{variable_name}={to_check}` to be present in `{check_in}`")

def quick_assert_valid_image(image_source, advise:str= "\nTry using setting `parse_image=True` or parse it yourself with `parse_image_as_uint8_rgb_numpy_array(image_source)`"):
    if not isinstance(image_source, ndarray):
        raise ValueError("`image_source` is not a numpy array. Expected a uint8 numpy image of shape (height, width, 3), but received something else." + advise)
    if len(image_source.shape) != 3:
        raise ValueError("`image_source` has an invalid shape. Expected a uint8 numpy image of shape (height, width, 3), but received something else." + advise)
    h, w, c = image_source.shape
    image_seems_ok = all([isinstance(image_source, np.ndarray), (image_source.dtype == "uint8"), (c == 3), (h > 3), (w > 3)])
    if not image_seems_ok:
        raise ValueError("`image_source` is not valid. Expected a uint8 numpy image of shape (height, width, 3), but received something else." + advise)

def assert_color(color:tuple, color_type:str, variable_name:str) -> None:
    """
    >> assert_color((200, 20, 200, 1.0), "RGBA")
    """
    assert_types([color, color_type, variable_name], [tuple, str, str], ["color", "color_type", "variable_name"])
    if color_type == "RGB":
        wrong_number_of_channels = len(color) != 3
        if wrong_number_of_channels:
            raise ValueError(f"`{variable_name}={color}` must have exactly 3 channels (RGB)")
        not_valid_rgb = not all((isinstance(c, int) and 0 <= c <= 255) for c in color)
        if not_valid_rgb:
            raise ValueError(f"`{variable_name}={color}` is not a legal RGB color. Expect a tuple with exactly 3 number between 0<=channel<=255")
    elif color_type == "RGBA":
        wrong_number_of_channels = (len(color) != 4)
        if wrong_number_of_channels:
            raise ValueError(f"`{color=}` must have exactly 4 channels (RGBA)")
        not_valid_rgb = not all((isinstance(c, int) and 0 <= c <= 255) for c in color[:3])
        not_valid_alpha = not (0.0<=color[3]<=1.0)
        if not_valid_rgb or not_valid_alpha:
            raise ValueError(f"`{variable_name}={color}` is not a legal RGBA color. Expect a tuple with exactly 4 number.\n"
                             "The first 3 should be RGB values in 0<=channel<=255 while the 4'th should be an alpha value in 0.0<alpha<1.0")
    else:
        raise RuntimeError(f"`{variable_name}={color}` is either not implemented or unexpected")


def assert_valid_universal_inputs(
        title:Optional[str]=None,
        display_image: bool = True,
        return_image: bool = True,
        max_output_image_size_wh: tuple = None,
        save_image_path:Optional[str] = None,
        resize_factor: float = 1.0,
        BGR2RGB: bool = False,
        parse_image:bool=True
) -> None:
    # Checks - types
    assert_types(
        [title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image],
        [str,   bool,          bool,         tuple,                    str,             float,         bool,    bool],
        ["title", "display_image", "return_image", "max_output_image_size_wh", "save_image_path", "resize_factor", "BGR2RGB", "parse_image"],
        [1,     0,             0,            1,                        1,               0,             0,       0]
    )

    # `title`
    if isinstance(title, str) and (len(title) == 0):
        msg = f"You have passed an empty '{title=}'. This will probably cause some unwanted behavior"
        warnings.warn(msg)

    # `resize_factor`, `parse_image`
    assert_resize_factor_is_ok(resize_factor)
    if (resize_factor != 1.0) and (not parse_image):
        warning_msg = f"Found non default `{resize_factor=}` together with `{parse_image=}`. " \
                      f"Can only resize when `parse_image=True`. Will set `resize_factor=1.0` "
        warnings.warn(warning_msg)

    # `display_image`, `return_image`, `save_image_path`
    if not any([display_image, return_image, save_image_path]):
        raise ValueError("If you don't want to display, return or save the image. What's the point :)")
    if save_image_path is not None:
        formats = global_config.image_formats
        assert any(save_image_path.lower().endswith(f) for f in formats),\
            f"`{save_image_path=}` has an unknown image format. Please use one of these: `{formats}`"

    # `max_output_image_size_wh`
    if max_output_image_size_wh is not None:
        assert len(max_output_image_size_wh) == 2, f"Expected `{max_output_image_size_wh=}` to contain exactly 2 numbers (width and height). None is allowed"
        w, h = max_output_image_size_wh
        w_is_not_none, h_is_not_none = (w is not None), (h is not None)

        if (w_is_not_none and w < 10) or (h_is_not_none and h < 10):
            raise ValueError(f"Expected both `{max_output_image_size_wh=}` to contain two numbers larger than 10 pixels.")
        if (w_is_not_none and w < 100) or (h_is_not_none and h < 100):
            msg = f"`{max_output_image_size_wh=}` is suspiciously small. Are you sure it's correct?"
            warnings.warn(msg)
        elif (w_is_not_none and w > 10_000) or (h_is_not_none and h > 10_000):
            msg = f"`{max_output_image_size_wh=}` is suspiciously large. Are you sure it's correct?"
            warnings.warn(msg)

def assert_positive_float(to_check:float, zero_allowed:bool=False, max_value_allowed:float=None, variable_name:str=None) -> None:
    assert_types([to_check, zero_allowed, max_value_allowed, variable_name],
                 [float, bool, float, str],
                 ["to_check", "zero_allowed", "max_value_allowed", "variable_name"],
                 [0, 0, 1, 1])

    assert (max_value_allowed is None) or (max_value_allowed >= 0.0), f"Expected max_value >= 0, but found `{max_value_allowed=}`"
    if max_value_allowed is None:
        max_value_allowed = np.inf

    if (not zero_allowed) and ( (to_check <= 0.0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0.0 < {to_check} <= {max_value_allowed}`")
    elif zero_allowed and ( (to_check < 0.0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0.0 <= {to_check} <= {max_value_allowed}`")

def assert_positive_int(to_check:int, zero_allowed:bool=False, max_value_allowed:int=None, variable_name:str=None) -> None:
    assert_types([to_check, zero_allowed, max_value_allowed, variable_name],
                 [int, bool, int, str],
                 ["to_check", "zero_allowed", "max_value_allowed", "variable_name"],
                 [0, 0, 1, 1])

    assert (max_value_allowed is None) or (max_value_allowed >= 0), f"Expected max_value >= 0, but found `{max_value_allowed=}`"
    if max_value_allowed is None:
        max_value_allowed = np.inf

    if (not zero_allowed) and ( (to_check <= 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 < {to_check} <= {max_value_allowed}`")
    elif zero_allowed and ( (to_check < 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 <= {to_check} <= {max_value_allowed}`")

def assumed_torch_batch(image_source) -> bool:
    # TODO: This was written fast, make sure the check is sufficient
    if not TORCH_FOUND:
        return False
    if not isinstance(image_source, torch.Tensor):
        return False
    if not (len(image_source.shape) == 4):
        return False


    # In the following checks I expect either shape
    # (batch_size, channel, height, width) or
    # (batch_size, height, width, channel)
    if not (image_source.shape[3] == 3 or image_source.shape[1] == 3):
        return False
    if not ((image_source.shape[1] > global_config.image_min_height) or (image_source.shape[2] > global_config.image_min_height)):
        return False
    if not ((image_source.shape[2] > global_config.image_min_width) or (image_source.shape[3] > global_config.image_min_width)):
        return False
    return True

def assumed_numpy_batch(image_source:ndarray) -> bool:
    # TODO: This was written fast, make sure the check is sufficient
    if not isinstance(image_source, ndarray):
        return False
    if not (len(image_source.shape) == 4):
        return False
    if not (image_source.shape[3] == 3):
        return False
    if not (image_source.shape[1] > global_config.image_min_height):
        return False
    if not (image_source.shape[2] > global_config.image_min_width):
        return False
    return True

def assert_valid_video_path(video_path:str) -> None:
    assert_path(video_path, "video_path")
    image_formats  = global_config.image_formats
    video_formats_all = global_config.video_formats_all
    video_formats_tested = global_config.video_formats_tested

    if any(video_path.endswith("." + image_format) for image_format in image_formats):
        raise ValueError(f"You have provided a string with an image extension: `{video_path}`. Legal video formats are:\n`{video_formats_all}`")
    elif not any(video_path.lower().endswith("." + video_format) for video_format in video_formats_all):
        raise ValueError(f"Unrecognized video format in `{video_path}`. Legal video formats are:\n`{video_formats_all}`")
    elif not any(video_path.lower().endswith("." + video_format) for video_format in video_formats_tested):
        video_format = video_path.split(".")[-1]
        warning_msg = f"The file `{video_path}` has a valid video extension: `{video_format}`. " \
                      f"However, this format has not been thoroughly tested, and unexpected errors may occur. " \
                      f"The currently tested video formats are:\n{video_formats_tested}"
        warnings.warn(warning_msg)

def assert_resize_factor_is_ok(resize_factor):
    assert_type(resize_factor, float, "resize_factor")
    if resize_factor <= 0.0:
        raise ValueError(f"`{resize_factor=}` is not valid, it must be strictly greater than zero")
    if resize_factor > 100.0:
        warning_msg = f"Encountered a suspiciously large value for `{resize_factor=}`"
        warnings.warn(warning_msg)

__all__ = [
    "assert_path",
    "assert_image_type",
    "assert_valid_image",
    "is_valid_url",
    "assert_type",
    "assert_types",
    "assert_list_types",
    "assert_dict_types",
    "assert_eval",
    "assert_in",
    "quick_assert_valid_image",
    "assert_color",
    "assert_valid_universal_inputs",
    "assert_positive_float",
    "assert_positive_int",
    "assumed_torch_batch",
    "assumed_numpy_batch",
    "assert_valid_video_path",
    "assert_resize_factor_is_ok"
]