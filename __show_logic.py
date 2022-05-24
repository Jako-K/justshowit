# I'm going to allow that torch is not installed, it seemed a unwise to demand the installation of such a large library
_TORCH_NOT_FOUND = False
try:
    import torch
except ModuleNotFoundError:
    _TORCH_NOT_FOUND = True

from PIL import Image
import requests
from rectpack import newPacker
import cv2
import numpy as np
import warnings as warnings
from . import __checker as checker
import os


def _get_collage_image(images:list, allow_rotations:bool=False):
    """
    Used to pack `images` into a single image.
    NOTE: This function is only intended to be used by `show()`

    @param images: list of images in np.ndarray format
    @param allow_rotations: Determine if the packing algorithm is allowed to rotate the images
    @return: A single collage image build from `images` in `np.ndarray` format
    """

    # A lot of the complexity is removed if all the images are of the same size. This means that a much more constrained approached can be used.
    if all([images[0].shape == image.shape for image in images]):
        cols, rows, resize_factor, _ = _get_grid_parameters(images)
        return _get_grid_image(images, cols, rows, resize_factor)

    # Setup
    rectangles = [(s.shape[0], s.shape[1], i) for i, s in enumerate(images)]
    height_domain = [60 * i for i in range(1, 8)] + [int(1080 * 5 / 1.05 ** i) for i in range(50, 0, -1)]
    width_domain = [100 * i for i in range(1, 8)] + [int(1920 * 5 / 1.05 ** i) for i in range(50, 0, -1)]
    canvas_dims = list(zip(height_domain, width_domain)) # Just different sizes to try
    canvas_image = None
    max_x, min_y = -1, 1e6 # Used to crop the grey parts

    # Attempt to pack all images into the smallest of the predefined width-height combinations
    for canvas_dim in canvas_dims:

        # Try packing
        packer = newPacker(rotation=allow_rotations)
        for r in rectangles: packer.add_rect(*r)
        packer.add_bin(*canvas_dim)
        packer.pack()

        # If all images couldn't fit, try with a larger image
        if len(packer.rect_list()) != len(images):
            continue

        # Setup
        canvas_image = np.zeros((canvas_dim[0], canvas_dim[1], 3)).astype(np.uint8) + 65 # 65 seems to be a pretty versatile grey color i.e. it looks decent no matter the pictures
        H = canvas_image.shape[0]

        for rect in packer[0]:
            image = images[rect.rid]
            h, w, y, x = rect.width, rect.height, rect.x, rect.y

            # Transform origin to upper left corner
            y = H - y - h

            # Handle image rotations if necessary
            if image.shape[:-1] != (h, w): image = image.transpose(1, 0, 2)
            canvas_image[y:y+h, x:x+w, :] = image

            if max_x < (x+w): max_x = x+w
            if min_y > y: min_y = y

        break

    if canvas_image is None:
        raise RuntimeError("Failed to produce mosaic image. This is probably caused by to many and/or to large images")

    if (max_x == -1) or (min_y == 1e6):
        raise RuntimeError("This should not be possible.")

    return canvas_image[min_y:, :max_x, :]


def _get_grid_parameters(images, max_height=1080, max_width=1920, desired_ratio=9/17):
    """
    Try at estimate #cols, #rows and resizing factor necessary for displaying a list of images in a visually pleasing way.
    This is essentially done by minimizing 3 separate parameters:
        (1) difference between `desired_ratio` and height/width of the final image
        (2) Amount of image scaling necessary
        (3) the number of empty cells (e.g. 3 images on a 2x2 --> empty_cell = 1)

    NOTE1: This was a pretty challenging function to write and the solution may appear a bit convoluted.
           I've included some notes at "doc/_get_grid_parameters.jpg" which will hopefully motivate the solution -
           in particular the loss-function used for optimization.
    NOTE2: This function is only intended to be used by `show()`

    @param images: list np.ndarray images
    @param max_height:
    @param max_width:
    @param desired_ratio:
    @return: cols, rows, scaling_factor, loss_info
    """

    N = len(images)
    h, w, _ = images[0].shape
    H, W = max_height, max_width

    losses = {}
    losses_split = []
    for a in [0.05 + 0.01 * i for i in range(96)]:
        for x in range(1, N + 1):
            for y in range(1, N + 1):

                # If the solution is not valid continue
                if (h * a * y > H) or (w * a * x > W) or (x * y < N):
                    continue
                # Otherwise calculate loss
                else:
                    ratio_loss = abs((h * y) / (w * x) - desired_ratio) # (1)
                    scale_loss = (1 - a) ** 2 # (2)
                    empty_cell_loss = x*y/N - 1 # (3)
                    losses[(y, x, a)] = ratio_loss + scale_loss + empty_cell_loss
                    losses_split.append([ratio_loss, scale_loss, empty_cell_loss])

    # pick parameters with the lowest loss
    rl, sl, ecl = losses_split[np.argmin(list(losses.values()))]
    loss_info = {"ratio":rl, "scale":sl, "empty_cell":ecl, "total":rl+sl+ecl}
    cols, rows, scaling_factor = min(losses, key=losses.get)
    return cols, rows, scaling_factor, loss_info


def _get_grid_image(images:list, cols:int, rows:int, resize_factor:float=1.0):
    """
    Put a list of np.ndarray images into a single combined image.
    NOTE: This function is only intended to be used by `show()`

    @param images: list np.ndarray images
    @param cols: Number of columns with images
    @param rows: Number of rows with images
    @param resize_factor: Image resize factor
    @return: On single picture with all the images in `images`
    """

    h_old, w_old, _ = images[0].shape
    h = int(h_old * resize_factor)
    w = int(w_old * resize_factor)
    canvas = np.zeros((int(h_old * resize_factor * cols), int(w_old * resize_factor * rows), 3))
    canvas = canvas.astype(np.uint8)

    image_index = -1
    for col in range(cols):
        for row in range(rows):
            image_index += 1
            if image_index >= len(images): break

            # Load and rescale image
            image = images[image_index]
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

            # Add image to the final image
            canvas[col * h: (col + 1) * h, row * w: (row + 1) * w, :] = image
    return canvas


def _get_image(source, resize_factor: float = 1.0, BGR2RGB: bool = None):
    """
    Take an image in format: path, url, ndarray or tensor.
    Returns `source` as np.ndarray image after some processing e.g. remove alpha

    NOTE: This function is only intended to be used by `show()`
    """

    # `source` and `resize` checks
    is_path = os.path.exists(source) if isinstance(source, str) else False
    is_url = True if isinstance(source, str) and checker.is_valid_url(source) is True else False
    is_ndarray = True if isinstance(source, np.ndarray) else False
    is_torch_tensor = True if (_TORCH_NOT_FOUND and isinstance(source, torch.Tensor)) else False

    if not any([is_path, is_url, is_ndarray, is_torch_tensor]):
        raise ValueError("`source` could not be interpreted as a path, url, ndarray or tensor.")

    if is_path and is_url:
        raise AssertionError("This should not be possible")  # Don't see how a `source` can be a path and a url simultaneously

    if resize_factor < 0:
        raise ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    if is_torch_tensor and len(source.shape) > 3:
        raise ValueError(f"Expected tensor image to be of shape (channels, height, width), but received: {source.shape}. "
                         f"If you passed a single image as a batch use <YOUR_IMAGE>.squeeze(0). "
                         f"Otherwise pick a single image or split the batch into individual images an pass them all")

    if is_torch_tensor and (len(source.shape) == 3) and (source.shape[0] >= source.shape[1] or source.shape[0] >= source.shape[2]):
        raise ValueError(f"Expect tensor image to be of shape (channels, height, width), but received: {source.shape}. "
                         f"If your image is of shape (height, width, channels) use `<YOUR_IMAGE>.permute(2, 0, 1)`")

    # Cast to Pillow image
    if is_path:
        image = Image.open(source)
    elif is_url:
        image = Image.open(requests.get(source, stream=True).raw)
    elif is_ndarray:
        image = Image.fromarray(source)
    elif is_torch_tensor:
        corrected = source.permute(1, 2, 0) if (len(source.shape) > 2) else source
        image = Image.fromarray(corrected.numpy())

    # Swap blue and red color channel stuff
    num_channels = len(image.getbands())
    bgr2rgb_auto = (BGR2RGB is None) and is_ndarray and (num_channels in [3, 4])
    if BGR2RGB or bgr2rgb_auto:
        # BGR --> RGB or BGRA --> RGBA
        as_array = np.asarray(image)
        color_corrected = cv2.cvtColor(as_array, cv2.COLOR_BGR2RGB) if (num_channels == 3) \
            else cv2.cvtColor(as_array, cv2.COLOR_BGRA2RGBA)
        image = Image.fromarray(color_corrected)

    if resize_factor != 1.0:
        width = int(image.size[0] * resize_factor)
        height = int(image.size[1] * resize_factor)
        image = image.resize((width, height), resample=0, box=None)

    image = np.asarray(image)

    # Adds 3 identical channels to greyscale images (for compatibility)
    if (len(image.shape) == 2) or (image.shape[-1] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Remove alpha channel (for compatibility)
    if image.shape[-1] == 4:
        image = image[:,:,:3]

    return image


def show(source, resize_factor:float=1.0, BGR2RGB:bool=False, return_image:bool=False, return_without_show:bool=False):
    """
    Display a single image or a list of images from path, np.ndarray, torch.Tensor or url.

    @param source: path, np.ndarray, torch.Tensor url pointing to the image you wish to display
    @param resize_factor: Rescale factor in percentage (i.e. between 0-1)
    @param BGR2RGB: Convert `source` from BGR to RGB. If `None`, will convert np.ndarray images automatically
    @param return_image: return image as `np.ndarray`
    @param return_without_show: In the off change that you need the image returned, but not shown.
                                NOTE: `return_image` must be True if `return_without_show` is True
    """

    # Checks
    if _TORCH_NOT_FOUND:
        checker.assert_in(type(source), [np.ndarray, str, list, tuple])
    else:
        checker.assert_in(type(source), [np.ndarray, str, torch.Tensor, list, tuple])
    checker.assert_types([resize_factor, BGR2RGB, return_without_show], [float, bool, bool], [0, 1, 0])
    if return_without_show and (not return_image):
        raise ValueError("`return_without_show` is not allowed to be True if `return_image` is False.")


    # Prepare the final image(s)
    if type(source) not in [list, tuple]:
        final_image = _get_image(source, resize_factor, BGR2RGB)
    else:
        if len(source) > 200: # Anything above 200 indexes seems unnecessary
            warnings.warn(
                f"Received `{len(source)}` images, the maximum limit is 200. "
                "Will pick 200 random images from `source` for display and discard the rest"
            )
            random_indexes_200 = np.random.choice(np.arange(len(source)), 200)
            source = [source[i] for i in random_indexes_200]

        images = [_get_image(image, resize_factor, BGR2RGB) for image in source]
        final_image = _get_collage_image(images, allow_rotations=False)


    # Resize the final image if it's larger than 2160x3840
    scale_factor = None
    if final_image.shape[1] > 3840:
        scale_factor = 3840/final_image.shape[1]
    elif final_image.shape[0] > 2160:
        scale_factor = 2160 / final_image.shape[0]
    if scale_factor:
        height = int(final_image.shape[0]*scale_factor)
        width = int(final_image.shape[1]*scale_factor)
        final_image = cv2.resize(final_image, (width, height))

    # Display and return image
    if checker.in_jupyter() and (not return_without_show):
        final_image = Image.fromarray(final_image)
        display(final_image)
        np.asarray(final_image)
    elif not return_without_show:
        cv2.imshow("Just show it", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_image if return_image else None

__all__=[
    "show"
 ]
