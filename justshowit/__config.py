from __future__ import annotations
from typing import Union, List, Tuple, Optional, Iterable, Dict
import numpy as np
import PIL
from PIL import Image
import warnings
import cv2

# A shared, global config file that will only be initialized ones, but that can be edited by the user.
# TODO: I should probably add a shit ton of setters that will warn the user if the input is invalid e.g. `show_max_image_amount<0`
class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    # --------------------------------------------
    # Image show and parse
    # --------------------------------------------

    # __show
    show_max_image_amount:int = 200
    image_min_width:int = 10 # This value may not go below 4, because of errors thrown in __parser
    image_min_height:int = 10  # This value may not go below 4, because of errors thrown in __parser

    # Torch
    try_inverse_torch_standardization:bool = True
    torch_standardization_mean:Tuple[int, int, int] = (0.485, 0.456, 0.406)
    torch_standardization_std:Tuple[int, int, int] = (0.229, 0.224, 0.225)

    # numpy
    image_normalization_clip_min:float = -0.01
    image_normalization_clip_max:float = 1.01
    image_clip_min:int = -1
    image_clip_max:int = 1

    # Video
    default_fixed_frame_count_video:int = 6

    # Grid optimization parameters for `show_grid`
    config_auto_grid_layout = {
        "desired_ratio":9/17, # The aspect ratio (h,w) the algorithm aims for
        "ratio_weight":1.0, # How highly difference between the actual and the desired aspect ratio is penalized
        "scale_weight":1.0, # How highly image resizing is penalized
        "empty_weight":1.0  # How highly empty grid cells are penalized
    }

    # --------------------------------------------
    # control
    # --------------------------------------------

    in_debug_mode:bool = False
    cv2_wait_key_delay_ms:int = 0 # 0.0 means it will wait indefinitely, any value greater than zero -> delay in millisecond

    # --------------------------------------------
    # Jupyter
    # --------------------------------------------

    # Check if python is currently running in a jupyter environment. NOTE: This code is somewhat dubious, but I have tested it to the best of my abilities.
    IN_JUPYTER = False
    try:
        shell = get_ipython().__class__.__name__  # This is supposed to be an unresolved reference anywhere outside jupyter
        if shell == 'ZMQInteractiveShell':
            IN_JUPYTER = True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            IN_JUPYTER = False  # Terminal running IPython (the check is unnecessary, but left it for future reference)
    except NameError:
        # Probably standard Python interpreter
        pass

    # --------------------------------------------
    # Image and video formats
    # --------------------------------------------

    video_formats_all:List[str] = ["avi", "mp4", "mkv", "mov", "wmv", "flv", "mpeg", "3gp", "webm", "asf", "vob", "ts", "m4v", "mpg", "rm", "swf", "m2ts", "ogg", "amv"]
    image_formats:list[str] = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    for image_format in image_formats:
        assert image_format not in video_formats_all, f"Overlap between video and image formats: `{image_format}`. This will results in problems later on."
    del image_format
    video_formats_tested:list[str] = ["mp4", "mov"]
    resize_interpolation:int = cv2.INTER_LINEAR #cv2.INTER_CUBIC

    # --------------------------------------------
    # Shared stuff
    # --------------------------------------------

    # So, yeah, this is admittedly an ugly solution.
    # But I want torch type hints e.g. `image:torch.Tensor` even if torch is not installed
    TORCH_FOUND = True
    try:
        import torch
    except ModuleNotFoundError:
        TORCH_FOUND = False

    if IN_JUPYTER:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    tqdm = tqdm

    if TORCH_FOUND:
        import torch
    else:
        class Torch:
            class dummy_torch_type_its_impossible_to_be_isinstance_off:
                pass
            Tensor = type(dummy_torch_type_its_impossible_to_be_isinstance_off)
        torch = Torch()

    if TORCH_FOUND:
        ImageType = Union[np.ndarray, torch.Tensor, PIL.Image.Image, str]
    else:
        ImageType = Union[np.ndarray, PIL.Image.Image, str]
    ImageSource = Union[np.ndarray, torch.Tensor, ImageType, List[ImageType]]

# Shared stuff
global_config:Config = Config()
ImageType = global_config.ImageType
ImageSource = global_config.ImageSource
ndarray = np.ndarray
torch = global_config.torch
tqdm = global_config.tqdm
TORCH_FOUND:bool = global_config.TORCH_FOUND
IN_JUPYTER:bool = global_config.IN_JUPYTER

__all__ = [
    "global_config", "Dict", "Union", "List", "Tuple", "Optional", "Iterable", "ImageType", "ImageSource", "ndarray", "tqdm", "warnings", "TORCH_FOUND", "IN_JUPYTER", "torch"
]