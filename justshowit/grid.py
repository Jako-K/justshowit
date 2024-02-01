from __future__ import annotations
import copy
import numpy as np
import cv2
import itertools
from PIL import Image, ImageDraw

from config import *
import show
import checker as checker
import image_modifier
import parsers
import utils

def _get_grid_dimensions(images:List[np.ndarray], rows:int, cols:int):
    # TODO: add return types
    coordinates = list(itertools.product(range(rows), range(cols)))
    coordinates = coordinates[:len(images)]
    grid_dims = np.zeros((rows, cols, 2))

    # Extract grid data
    for i, (row, col) in enumerate(coordinates):
        image_height, image_width = images[i].shape[:2]

        # Max col/row shape
        grid_dims[row, col, 0] = image_width
        grid_dims[row, col, 1] = image_height

    # Find the per columns/row maximum width/height
    max_height_per_row = grid_dims[:, :, 1].max(axis=1)
    assert len(max_height_per_row) == rows
    max_width_per_col = grid_dims[:, :, 0].max(axis=0)
    assert len(max_width_per_col) == cols

    return coordinates, max_width_per_col, max_height_per_row

def _get_canvas(dummy_draw, row_text, col_text, row_font, col_font, max_height_per_row, rows, row_spacing, text_adjusted_font_size, max_width_per_col, cols, col_spacing, canvas_background_color, image_area_margins):
    # Find out how much extra width/height is needed for row/col text
    extra_height_col_text, extra_width_row_text = 0, 0
    if row_text:
        extra_width_row_text =  int(max([dummy_draw.textbbox((0, 0), text, font=row_font)[2] for text in row_text]) * 1.2)
    if col_text:
        extra_height_col_text = int(max([dummy_draw.textbbox((0, 0), text, font=col_font)[3] for text in col_text]) * 1.2)

    # Margins
    margins_width = image_area_margins["left"] + image_area_margins["right"]
    margins_height = image_area_margins["top"] + image_area_margins["bottom"]

    # Construct the canvas find a suitable size given the inputs
    canvas_height = margins_height + extra_height_col_text + sum(max_height_per_row) + (rows-1)*row_spacing + (rows-1)*text_adjusted_font_size
    canvas_width =  margins_width  + extra_width_row_text  + sum(max_width_per_col)  + (cols-1)*col_spacing
    canvas_height, canvas_width = int(canvas_height), int(canvas_width)
    canvas = np.zeros((canvas_height, canvas_width, 3)).astype(np.uint8)
    canvas[...] = canvas_background_color
    return canvas, extra_height_col_text, extra_width_row_text


# noinspection PyTypeChecker
def _draw_text(canvas, x, y, text, color, font):
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), text, color, font=font)
    canvas = np.array(canvas)
    return canvas

def _get_cell_data(images, image_text, offset_x, offset_y, coordinates, dummy_draw, text_font, col_spacing, max_width_per_col, row_spacing, text_adjusted_font_size, max_height_per_row):
    cell_data = []

    for i, (row, col) in enumerate(coordinates):
        image = images[i]
        image_height, image_width, _ = image.shape
        cell_text, text_width, text_height = None, 0, 0
        if image_text is not None:
            cell_text = image_text[i]
            _, _, text_width, text_height = dummy_draw.textbbox((0, 0), cell_text, font=text_font)

        # Cell coordinates
        cell_upper_left_x = offset_x + sum(max_width_per_col[:col])  + col*col_spacing
        cell_upper_left_y = offset_y + sum(max_height_per_row[:row]) + row*(row_spacing+text_adjusted_font_size)
        cell_lower_right_x = cell_upper_left_x + max_width_per_col[col]
        cell_lower_right_y = cell_upper_left_y + max_height_per_row[row]

        # Image coordinates (centered within the cell width)
        image_upper_left_x = cell_upper_left_x + (max_width_per_col[col] - image_width)/2
        image_upper_left_y = cell_upper_left_y
        image_lower_right_x = image_upper_left_x + image_width
        image_lower_right_y = image_upper_left_y + image_height

        # Center of the cell
        cell_center_x = cell_upper_left_x + (cell_lower_right_x - cell_upper_left_x)/2
        cell_center_y = cell_upper_left_y + (cell_lower_right_y - cell_upper_left_y)/2

        assert int(image_lower_right_x - image_upper_left_x) == image_width,  "The calculated coordinates doesn't match with image's width"
        assert int(image_lower_right_y - image_upper_left_y) == image_height, "The calculated coordinates doesn't match with image's height"

        cell_data.append({
            "row":row,
            "col":col,
            "image": image,
            "image_height":image_height,
            "image_width":image_width,
            "cell_text":cell_text,
            "text_width":text_width,
            "text_height":text_height,
            "cell_upper_left_x":  int(cell_upper_left_x),
            "cell_upper_left_y":  int(cell_upper_left_y),
            "cell_lower_right_x":  int(cell_lower_right_x),
            "cell_lower_right_y":  int(cell_lower_right_y),
            "image_upper_left_x": int(image_upper_left_x),
            "image_upper_left_y": int(image_upper_left_y),
            "image_lower_right_x": int(image_lower_right_x),
            "image_lower_right_y": int(image_lower_right_y),
            "cell_center_x":      int(cell_center_x),
            "cell_center_y":      int(cell_center_y),
        })
    return cell_data

def _draw_separation_lines(canvas, sep_line_color, image_area_offset_y, rows, row_spacing, text_adjusted_font_size, max_height_per_row, image_area_offset_x, cols, col_spacing, max_width_per_col, cell_data, col_sep_line, row_sep_line):
    # If the line is this faint, I might as well just skip the line drawing all together, right?
    if sep_line_color[3] < 0.01:
        return canvas

    alpha_overlay = canvas.copy() if (sep_line_color and (sep_line_color[3] < 1.0)) else None
    y_min = int(image_area_offset_y)
    y_max = int(image_area_offset_y + (rows - 1) * (row_spacing + text_adjusted_font_size) + max_height_per_row.sum())
    x_min = int(image_area_offset_x)
    x_max = int(image_area_offset_x + (cols - 1) * col_spacing + max_width_per_col.sum())

    for cell in cell_data:
        x2, y2 = cell["cell_lower_right_x"], cell["cell_lower_right_y"]

        if (cell["row"] == 0) and (cell["col"] < (cols - 1)) and (col_sep_line > 0):
            x = round(x2 + col_spacing / 2 - round(col_sep_line / 2))  # The double round just looks better for whatever reason
            cv2.line(canvas, (x, y_min), (x, y_max), sep_line_color, col_sep_line)

        if (cell["col"] == 0) and (cell["row"] < (rows - 1)) and (row_sep_line > 0):
            y = round(y2 + row_spacing / 2 - round(row_sep_line / 2) + text_adjusted_font_size / 2)  # The double round just looks better for whatever reason
            cv2.line(canvas, (x_min, y), (x_max, y), sep_line_color, row_sep_line)

    # Line transparency
    if alpha_overlay is not None:
        alpha = sep_line_color[3]
        canvas = cv2.addWeighted(alpha_overlay, 1 - alpha, canvas, alpha, 0)

    return canvas

def _resize_images(images: List[np.ndarray], config: Dict):
    """
    So this is both inefficient and kinda ugly, and I'm unapologetic about it :)
    The thing is, Firstly, I don't want to have to write however many combinations there is for different values of
    [min_width, min_height, max_width, max_height]
    Secondly, I would really like to avoid messing about with the aspect ratio if at all possible.
    So I first attempt a clumsy, but aspect respecting resize and fall back on brute force resize if that fails
    """

    # Just return the images as is if nothing needs to be done
    if not any([config["max_width"], config["max_height"], config["min_width"], config["min_height"]]):
        return images

    # Setup
    max_width = config["max_width"] if config["max_width"] else 1e6
    max_height = config["max_height"] if config["max_height"] else 1e6
    min_width = config["min_width"]
    min_height = config["min_height"]
    interpolation = config["interpolation"]

    # Resize all images
    for i, image in enumerate(images):
        h, w, _ = image.shape

        # Attempt resize without altering the aspect ratio
        if min_width and (min_width > w):
            image, h, w = image_modifier.resize_image_respect_aspect_ratio(image, min_width, None, interpolation)
        if min_height and (min_height > h):
            image, h, w = image_modifier.resize_image_respect_aspect_ratio(image, None, min_height, interpolation)
        if max_width and (max_width < w):
            image, h, w = image_modifier.resize_image_respect_aspect_ratio(image, max_width, None, interpolation)
        if max_height and (max_height < h):
            image, h, w = image_modifier.resize_image_respect_aspect_ratio(image, None, max_height, interpolation)

        # If all else fail, ignore the aspect ratio and ensure the resize respect the users input
        if (h < min_height) or (h > max_height) or (w < min_width) or (w > max_width):
            width = min_width if (w < min_width) else min(w, max_width)
            height = min_height if (h < min_height) else min(h, max_height)
            image = cv2.resize(image, (width, height), interpolation=interpolation)
        images[i] = image
    return images

def _add_padding(images, border_size, color, add_padding_on_the_outside):
    if not border_size:
        return images

    s = border_size
    for i, image in enumerate(images):
        if add_padding_on_the_outside:
            padded_image = cv2.copyMakeBorder(image, s, s, s, s, cv2.BORDER_CONSTANT, value=color)
        else:
            padded_image = np.full(image.shape, color, dtype=np.uint8)
            padded_image[s:-s, s:-s, :] = image[s:-s, s:-s, :]
        images[i] = padded_image
    return images

def _check_dict(image_spacing:dict, image_resizing:dict, image_text_config:dict, col_text_config:dict, row_text_config:dict, image_area_margins:dict, image_border:dict, drop_down_shadow:dict):

    # --------------------------------------------------------------------
    # Checks - `image_spacing`
    # --------------------------------------------------------------------

    if image_spacing.get("sep_line_color"):
        checker.assert_color(image_spacing["sep_line_color"], "RGBA", 'image_spacing["sep_line_color"]')

    checker.assert_dict_types(
        dict_name="image_spacing",
        dict_to_check=image_spacing,
        names=["col_spacing", "row_spacing", "col_sep_line", "row_sep_line", "sep_line_color"],
        types=[int, int, int, int, tuple],
        must_be_present=[1, 1, 1, 1, 1],
        allow_nones=[0, 0, 0, 0, 0],
        eval_expr=["0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6", None],
        assert_dict_lte=True
    )

    # `col_spacing`, `row_spacing`
    checker.assert_eval(image_spacing["col_spacing"], "0<=x", False, "col_spacing")
    checker.assert_eval(image_spacing["row_spacing"], "0<=x", False, "row_spacing")

    # `col_sep_line`, `row_sep_line`
    if image_spacing["col_sep_line"]:
        assert image_spacing["col_sep_line"] < (image_spacing["row_spacing"] + 2)
    if image_spacing["row_sep_line"]:
        assert image_spacing["row_sep_line"] < (image_spacing["col_spacing"] + 2)

    # `canvas_background_color`, `sep_line_color`
    if image_spacing["sep_line_color"]:
        checker.assert_color(image_spacing["sep_line_color"], "RGBA", 'image_spacing["sep_line_color"]')

    # --------------------------------------------------------------------
    # Checks - `image_resize`
    # --------------------------------------------------------------------

    if image_resizing is not None:
        if image_resizing.get("interpolation") and not (0 <= image_resizing["interpolation"] <= 7):
            raise ValueError("As of writing this, there's 8 valid interpolation in cv2. These are encoded with an enum spanning from 0-7 (both ends included)."
                             "And include the following names (`cv2.` prefix):"
                             "[INTER_NEAREST, INTER_LINEAR, INTER_LINEAR_EXACT, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_NEAREST_EXACT, INTER_MAX]")
        checker.assert_dict_types(
            dict_name="image_resizing",
            dict_to_check=image_resizing,
            names=["max_width", "max_height", "min_width", "min_height", "interpolation"],
            types=[int, int, int, int, int],
            must_be_present=[0, 0, 0, 0, 0],
            allow_nones=[1, 1, 0, 0, 0],
            eval_expr=["0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6", "0<=x<=7"],
            assert_dict_lte=True
        )

    # --------------------------------------------------------------------
    # Checks - `image_text_config`, `col_text_config`, `row_text_config`
    # --------------------------------------------------------------------

    # TODO: This is what I want to have implemented
    # to_check = [
    #     (image_text_config, "image_text_config", ['title_left', 'title_middle', 'left', 'middle']),
    #     (col_text_config, "col_text_config", ['title_left', 'title_middle', 'left', 'middle']),
    #     (row_text_config, "row_text_config", ['right'])
    # ]
    # This is what is actually implemented
    to_check = [
        (image_text_config, "image_text_config",['title_middle']),
        (col_text_config, "col_text_config",    ['title_middle']),
        (row_text_config, "row_text_config",    ['right'])
    ]

    for (dict_to_check, dict_name, legal_placements) in to_check:
        if dict_to_check is None:
            continue

        # Manual checks
        if dict_to_check.get("color"):
            checker.assert_color(dict_to_check["color"], "RGB", 'dict_to_check["color"]')
        if dict_to_check.get("placement"):
            checker.assert_in(dict_to_check["placement"], legal_placements, f"{dict_name}['placement']")
        if dict_to_check.get("font_thickness") is not None:
            if not (0<dict_to_check["font_thickness"]<=7):
                raise ValueError(f"`{dict_name}['font_thickness']={dict_to_check['font_thickness']}` is not valid. As of writing this, `font_thickness` must be in [1,2,3,4,5,6,7]")

        checker.assert_dict_types(
            dict_name=dict_name,
            dict_to_check=dict_to_check,
            names=['placement', 'font_size', 'font_thickness', 'italic', 'color', 'adjust_draw_distance'],
            types=[str, int, int, bool, tuple, (str, int)],
            must_be_present=[1, 1, 1, 1, 1, 1],
            allow_nones=[0, 0, 0, 0, 0, 1],
            eval_expr=[None, "0<=x<=1e6", "0<x<=7", None, None, None],
            assert_dict_lte=True
        )

    # --------------------------------------------------------------------
    # Checks - `image_area_margins`
    # --------------------------------------------------------------------

    if image_area_margins is not None:
        checker.assert_dict_types(
            dict_name="image_area_margins",
            dict_to_check=image_area_margins,
            names=["left", "right", "top", "bottom"],
            types=[int, int, int, int],
            must_be_present=[1, 1, 1, 1],
            allow_nones=[0, 0, 0, 0],
            eval_expr=["0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6", "0<=x<=1e6"],
            assert_dict_lte=True
        )

    # --------------------------------------------------------------------
    # Checks - `image_border`
    # --------------------------------------------------------------------

    if image_border is not None:
        if image_border.get("color"):
            checker.assert_color(image_border["color"], "RGB", 'image_border["color"]')
        checker.assert_dict_types(
            dict_name="image_border",
            dict_to_check=image_border,
            names=["border_size", "color", "add_padding_on_the_outside"],
            types=[int, tuple, int],
            must_be_present=[1, 1, 1],
            allow_nones=[0, 0, 0],
            eval_expr=["0<=x<=1e6", None, None],
            assert_dict_lte=True
        )

    # --------------------------------------------------------------------
    # Checks - `drop_down_shadow`
    # --------------------------------------------------------------------

    if drop_down_shadow is not None:
        if drop_down_shadow.get("color"):
            checker.assert_color(drop_down_shadow["shadow_color"], "RGB", 'drop_down_shadow["shadow_color"]')
        checker.assert_dict_types(
            dict_name="drop_down_shadow",
            dict_to_check=drop_down_shadow,
            names=["shadow_offset_x", "shadow_offset_y", "shadow_color", "iterations", "blur_kernel", "shadow_margins_lrtb"],
            types=[int, int, tuple, int, tuple, tuple],
            must_be_present=[1, 1, 1, 1, 1, 1],
            allow_nones=[0, 0, 0, 0, 0, 0],
            eval_expr=["x<=1e6", "x<=1e6", None, "0<=x<=1e6", "len(x)==2", "len(x)==4"],
            assert_dict_lte=True
        )
        if any([((k % 2 == 0) or (0 > k)) for k in drop_down_shadow["blur_kernel"]]):
            raise ValueError(f"`{drop_down_shadow['blur_kernel']=}` is not valid. It should contain exactly 2 odd, strictly positive numbers")
        if drop_down_shadow["iterations"] > 10:
            msg = f"{drop_down_shadow['iterations']=} seems unnecessarily high. The blur effect is pretty expensive."
            warnings.warn(msg)
        if any(0 > n for n in drop_down_shadow["shadow_margins_lrtb"]):
            raise ValueError(f"`{drop_down_shadow['shadow_margins_lrtb']=}` is not valid. All margins should be non-negative")

def _attempt_to_determine_rows_and_or_cols(rows, row_text, cols, col_text, num_images):
    if (row_text is not None) and (rows is None):
        raise ValueError("You have provided `row_text` but not `rows`. Please provide both or none.")
    if (col_text is not None) and (cols is None):
        raise ValueError("You have provided `col_text` but not `cols`. Please provide both or none.")

    if (rows is None) and (cols is None):
        for rows in range(100):
            cols = int(rows * (16/9))
            if (rows * cols) >= num_images:
                break
        if ((rows-1) * cols) >= num_images:
            rows = rows - 1
    elif (rows is None) and (cols is not None):
        rows = int(np.ceil(num_images/cols))
    elif (rows is not None) and (cols is None):
        cols = int(np.ceil(num_images / rows))

    failed_auto_detect = (not cols) or (not rows) or (rows*cols < num_images)
    if failed_auto_detect:
        raise ValueError("Failed to automatically determine `rows` and `cols` please provide both manually")
    return rows, cols

def _get_font_grid(text, placement: str, font_size: int, font_thickness: int, italic: bool, color, adjust_draw_distance: int):
    if text is None:
        return None, 0

    font = utils.get_font(font_size, font_thickness, italic)

    # Adjusted font size
    adjusted_font_size = font_size if ("title" in placement) else 0
    if adjust_draw_distance is None:
        adjusted_font_size += max(5, int(font_size / 3))
    else:
        adjusted_font_size += adjust_draw_distance
    return font, adjusted_font_size

# noinspection PyDefaultArgument
def show_grid_configurable(
        image_source: ImageSource,
        rows: int = None,
        cols: int = None,
        image_text: List[str] = None,
        col_text: List[str] = None,
        row_text: List[str] = None,
        canvas_background_color: Tuple[int, int, int] = (255, 255, 255),
        title: Optional[str] = None,
        display_image: bool = True,
        return_image: bool = False,
        max_output_image_size_wh: Optional[tuple] = (1920, 1080),
        save_image_path: Optional[str] = None,
        resize_factor: float = 1.0,
        BGR2RGB: bool = False,
        parse_image: bool = True,
        c_image_spacing:      Optional[Dict] = {"col_spacing":5, "row_spacing":5, "col_sep_line":1, "row_sep_line":1, "sep_line_color":(175, 175, 175, 0.3)},
        c_image_area_margins: Optional[Dict] = {"left": 40, "right":40, "top":40, "bottom":40},
        c_image_resizing:     Optional[Dict] = None,
        c_image_text_config:  Optional[Dict] = {'placement': 'title_middle', 'font_size': 15, 'font_thickness': 3, 'italic': False, 'color': (75, 75, 75), 'adjust_draw_distance': None},
        c_col_text_config:    Optional[Dict] = {"placement": "title_middle", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance":40},
        c_row_text_config:    Optional[Dict] = {"placement": "right", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance":40},
        c_image_border:       Optional[Dict] = None,
        c_drop_down_shadow:   Optional[Dict] = None,
) -> Optional[ndarray]:
    # TODO: Implement the different placements. (DONE, but not tested) Or at a minimum throw a NotImplementedError
    """

    Display a grid of images from `image_source` with specified number of `rows` and `cols`.
    NOTE: arguments prefixed with `c_` are dicts with extra configuration details. See examples below

    :param image_source: Two or more images. Can handle most image sources: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param rows: The number of rows in the grid. If not specified, it is calculated automatically.
    :param cols: The number of columns in the grid. If not specified, it is calculated automatically.
    :param image_text: List of texts to be displayed above each image.
    :param col_text: List of texts to be displayed at the top of each column.
    :param row_text: List of texts to be displayed on the right side of each row.
    :param canvas_background_color: Background color of the canvas in RGB format.
    :param title: Custom title for the displayed frames. If not provided, the video file name will be used as the title.
    :param display_image: If True, display the processed image using cv2.imshow() or PIL if in jupyter.
    :param return_image: If True, return the processed image as a np.ndarray (H, W, 3).
    :param max_output_image_size_wh: Maximum width and height of the displayed/returned image. If None, nothing will be done.
    :param save_image_path: If provided, save the processed image at the given file path. Default is None (no saving).
    :param resize_factor: Resize factor for each image/frame. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB before display/return.
    :param parse_image: If True, will thoroughly check image_source and ensure it's in the correct format
    :param c_image_spacing: Configuration for image spacing, margins, and separation lines. If None, nothing will be applied.
    Examples: {"col_spacing":40, "row_spacing":30, "col_sep_line":1, "row_sep_line":1, "sep_line_color":(175, 175, 175, 0.3)}
    :param c_image_area_margins: Configuration for image area margins. If None, nothing will be applied.
    Example: {"left": 40, "right":40, "top":40, "bottom":40}`
    :param c_image_resizing: Configuration for image resizing. If None, nothing will be applied.
    Example: {"max_width":800, "max_height":500, "min_width":0, "min_height":0, "interpolation":cv2.INTER_CUBIC}
    :param c_image_text_config: Configuration for image text. If None, nothing will be applied.
    Example: {"placement": "title_middle", "font_size": 15, "font_thickness": 3, "italic": False, "color": (75, 75, 75), "adjust_draw_distance":None}
    :param c_col_text_config: Configuration for column text. If None, nothing will be applied.
    Example: {"placement": "title_middle", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance":40}
    :param c_row_text_config: Configuration for row text. If None, nothing will be applied.
    Example: {"placement": "right", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance":40}
    :param c_image_border: Configuration for image border. If None, nothing will be applied.
    Example: {"border_size":1, "color":(200, 200, 200), "add_padding_on_the_outside":False}
    :param c_drop_down_shadow: Configuration for drop-down shadow. If None, nothing will be applied.
    :returns: If return_image is True, return an image of shape np.ndarray (H, W, 3), otherwise None
    """

    # --------------------------------------------------------------------
    # # Universal stuff and image parsing
    # --------------------------------------------------------------------

    checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image)

    not_iterable = not isinstance(image_source, (tuple, list))
    is_tensor = (TORCH_FOUND and isinstance(image_source, torch.Tensor))
    is_numpy = isinstance(image_source, np.ndarray)
    general_invalid_msg = "Please provide one of the following:\n" \
                          "(1) a batch np.ndarray images (N, height, width, 3), a torch.Tensor batch or\n" \
                          "(2) a batch of torch.Tensor images (N, 3, height, width)\n" \
                          "(3) a list of images comprised of (np.ndarray, torch.Tensor, PIL.Image.Image, url:str, path:str)"
    if (not_iterable and (not is_tensor) and (not is_numpy)) or (image_source is None):
        raise ValueError(f"`{type(image_source)=}` is not valid."+general_invalid_msg)
    elif not_iterable: # TODO test batch loading
        if (is_numpy or is_tensor) and (not parse_image):
            msg = f"`{parse_image=}` can only be used together with a iterable `type(image_source)`. Will continue parse_image=False"
            warnings.warn(msg)
        if isinstance(image_source, torch.Tensor) and checker.assumed_torch_batch(image_source):
            image_source = parsers.parse_torch_image_batch_as_uint8_rgb_numpy_array(image_source, resize_factor, BGR2RGB)
        elif isinstance(image_source, np.ndarray) and checker.assumed_numpy_batch(image_source):
            image_source = parsers.parse_numpy_image_batch_as_uint8_rgb_numpy_array(image_source, resize_factor, BGR2RGB)
        else:
            raise ValueError(f"You have provided a `{type(image_source)=}` but could not parse it as a batch of images."+general_invalid_msg)
    else:
        if parse_image:
            for i, image in enumerate(image_source):
                    image_source[i] = parsers.parse_image_as_uint8_rgb_numpy_array(image, resize_factor, BGR2RGB)
        else:
            image_source = parsers.quick_parse_image_source(image_source)

    images = image_source # A bit cumbersome to carry around, but I wanted to keep `image_source` for the sake of consistency between show-functions
    assert len(images) >= 2, f"Expected at least two images, but received `{len(images)=}`"

    max_amount_reached = (len(images) > global_config.show_max_image_amount)
    if max_amount_reached:
        if (rows is None) and (cols is None):
            images = parsers.threshold_image_count(images)
        else:
            msg = f"`{len(images)=}` exceeds the limit `{global_config.show_max_image_amount}`, but the program will continue as is because rows/cols are specified."
            warnings.warn(msg)


    # --------------------------------------------------------------------
    # Checks - Types
    # --------------------------------------------------------------------

    # Type checks
    checker.assert_types(
        [rows,   cols,   image_text,   col_text,   row_text,     canvas_background_color,   c_image_spacing,   c_image_resizing, c_image_text_config, c_col_text_config, c_row_text_config, c_image_area_margins, c_image_border, c_drop_down_shadow],
        [int,    int,    list,         list,       list,         tuple,                     dict,              dict,             dict,                dict,              dict,              dict,                 dict,           dict],
        ["rows", "cols", "image_text", "col_text", "row_text",   "canvas_background_color", "c_image_spacing", "image_resizing", "image_text_config", "col_text_config", "row_text_config", "image_area_margins", "image_border", "drop_down_shadow"],
        [True,   True,   True,         True,       True,         False,                     True,              True,             True,                True,              True,              True,                 True,           True]
    )

    # List type checks
    if image_text: checker.assert_list_types(image_text, str, "image_text")
    if col_text: checker.assert_list_types(col_text, str, "col_text")
    if row_text: checker.assert_list_types(row_text, str, "row_text")

    # --------------------------------------------------------------------
    # Copy mutable default arguments to avoid any monkey business
    # --------------------------------------------------------------------
    # Note: Mutable defaults are chosen because the user can directly see what options can be used

    c_image_spacing = copy.deepcopy(c_image_spacing)
    c_image_area_margins = copy.deepcopy(c_image_area_margins)
    c_image_resizing = copy.deepcopy(c_image_resizing)
    c_image_text_config = copy.deepcopy(c_image_text_config)
    c_col_text_config = copy.deepcopy(c_col_text_config)
    c_row_text_config = copy.deepcopy(c_row_text_config)
    c_image_border = copy.deepcopy(c_image_border)
    c_drop_down_shadow = copy.deepcopy(c_drop_down_shadow)

    # --------------------------------------------------------------------
    # Assign default values if necessary
    # --------------------------------------------------------------------

    # Text will be set to zero height
    if image_text is None:
        text_adjusted_font_size = 0

    # Margins will all be set to zero
    if c_image_area_margins is None:
        c_image_area_margins = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    # All spacing will be set to zero if None
    default_image_spacing = {"col_spacing": 0, "row_spacing": 0, "col_sep_line": 0, "row_sep_line": 0, "sep_line_color": (0, 0, 0, 1.0)}
    if c_image_spacing is None:
        c_image_spacing = default_image_spacing

    # If only some of the resize parameters are specified, default the rest
    if (c_image_resizing is not None) and (len(c_image_resizing) != 5):
        image_resizing_temp = {"max_width":int(1e6), "max_height":int(1e6), "min_width":0, "min_height":0, "interpolation":cv2.INTER_CUBIC}
        for k, v in c_image_resizing.items(): image_resizing_temp[k]=v
        c_image_resizing = image_resizing_temp; del image_resizing_temp

    # If the user has defined some variable, but not all, the remaining ones will get assigned default values
    for k, v in c_image_spacing.items():
        if c_image_spacing.get(k) is None:
            c_image_spacing[k] = default_image_spacing[k]

    # If the user haven't specified both `cols` and `rows` they will be chosen automatically
    if (rows is None) or (cols is None):
        rows, cols = _attempt_to_determine_rows_and_or_cols(rows, row_text, cols, col_text, num_images=len(images))

    # --------------------------------------------------------------------
    # Checks - Values
    # --------------------------------------------------------------------

    checker.assert_color(canvas_background_color, "RGB", "canvas_background_color")

    if not any([display_image, return_image, save_image_path]):
        raise ValueError("If you don't want to display, return or save the image. What's the point :)")

    # Dict checks (this is primarily type checks, but also some value checks)
    _check_dict(c_image_spacing, c_image_resizing, c_image_text_config, c_col_text_config, c_row_text_config, c_image_area_margins, c_image_border, c_drop_down_shadow)

    # These variables are littered all over the place, so I don't want to have to access them through a dict everytime
    col_spacing = c_image_spacing["col_spacing"]
    row_spacing = c_image_spacing["row_spacing"]
    col_sep_line = c_image_spacing["col_sep_line"]
    row_sep_line = c_image_spacing["row_sep_line"]
    sep_line_color = c_image_spacing["sep_line_color"]

    # `rows`, `cols`
    assert (rows > 0) and (cols > 0), f"Both `{rows=}` and `{cols=}` must be greater then 1"
    assert (rows * cols) >= len(images), f"You have provided too few `{cols=}` and/or `{rows=}`. `{rows*cols=}` is less then `{len(images)=}`"

    # `image_text`
    if image_text is not None:
        assert len(image_text) == len(images), f"Mismatch between the number of images `{len(images)=}` and the number of image texts `{len(image_text)=}`\n" \
                                               f"Provide a matching number or set `image_text=None`"
        if not all(len(t.splitlines()) <= 1 for t in image_text):
            msg = "Multiline strings are not supported because they cause all sorts of issues with the image layout." \
                  "\nChange the following entries in `image_text` to resolve the issue:\n"
            msg += ".\n".join([repr(t) for t in image_text if (len(t.splitlines()) != 1)])
            raise ValueError(msg)

    # `col_text` `row_text`
    if (col_text is not None) and (len(col_text) != cols):
        raise ValueError(f"There's a mismatch between the specified `{cols=}` and the number of `{len(col_text)=}`")
    if (row_text is not None) and (len(row_text) != rows):
        raise ValueError(f"There's a mismatch between the specified `{rows=}` and the number of `{len(row_text)=}`")

    #`image_text`, `image_text_config`, `col_text`, `col_text_config`, `row_text`, `row_text_config`
    if image_text and (not c_image_text_config):
        template = str({"placement": "title_middle", "font_size": 15, "font_thickness": 3, "italic": False, "color": (75, 75, 75), "adjust_draw_distance": None})
        raise ValueError(f"You must provide a valid `image_text_config` together with your `image_text`. Here a functional template:\nimage_text_config={template}")
    if col_text and (not c_col_text_config):
        template = str({"placement": "middle", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance": 70})
        raise ValueError(f"You must provide a valid `col_text_config` together with your `col_text`. Here a functional template:\ncol_text_config={template}")
    if row_text and (not c_row_text_config):
        template = str({"placement": "right", "font_size": 22, "font_thickness": 5, "italic": False, "color": (75, 75, 75), "adjust_draw_distance": 40})
        raise ValueError(f"You must provide a valid `row_text_config` together with your `row_text`. Here a functional template:\rows_text_config={template}")

    # `canvas_background_color`, `sep_line_color`
    checker.assert_color(canvas_background_color, "RGB", 'canvas_background_color')

    # --------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------

    # Text
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
    text_font, col_font, row_font = None, None, None
    col_adjusted_font_size, row_adjusted_font_size = None, None
    if image_text:
        text_font, text_adjusted_font_size = _get_font_grid(image_text, **c_image_text_config)
    if col_text:
        col_font, col_adjusted_font_size = _get_font_grid(col_text, **c_col_text_config)
    if row_text:
        row_font, row_adjusted_font_size = _get_font_grid(row_text, **c_row_text_config)
    # TODO: Add warnings for if the text exceeds the maximum height and or width for it's column/row
    # TODO: Add warnings if the image_text exceeds the boundaries of its cell

    # Reformat, resize and padding
    if c_image_border:
        images = _add_padding(images, **c_image_border)
    if c_image_resizing:
        images = _resize_images(images, c_image_resizing)

    # Grid and canvas
    coordinates, max_width_per_col, max_height_per_row = _get_grid_dimensions(images, rows, cols)
    canvas, extra_height_col_text, extra_width_row_text = _get_canvas(dummy_draw, row_text, col_text, row_font, col_font, max_height_per_row, rows, row_spacing, text_adjusted_font_size, max_width_per_col, cols, col_spacing, canvas_background_color, c_image_area_margins)
    image_area_offset_x, image_area_offset_y = extra_width_row_text + c_image_area_margins["left"], extra_height_col_text + c_image_area_margins["top"]
    cell_data = _get_cell_data(images, image_text, image_area_offset_x, image_area_offset_y, coordinates, dummy_draw, text_font, col_spacing, max_width_per_col, row_spacing, text_adjusted_font_size, max_height_per_row)

    # --------------------------------------------------------------------
    # Draw image area
    # --------------------------------------------------------------------

    # Dropdown shadow
    if c_drop_down_shadow is not None:
        (left, right, top, bottom) = c_drop_down_shadow.pop("shadow_margins_lrtb")
        shadow_upper_left_corner_xy = (image_area_offset_x - left, image_area_offset_y - top)
        shadow_lower_right_corner_xy = (canvas.shape[1] - c_image_area_margins["right"] + right, canvas.shape[0] - c_image_area_margins["bottom"] + bottom)
        canvas = image_modifier.add_square_dropdown_shadow(canvas, shadow_upper_left_corner_xy, shadow_lower_right_corner_xy, background_color=canvas_background_color, **c_drop_down_shadow)
        # TODO add warning if the the shadow margins exceed the shape of the canvas

    # Draw separation lines between images
    if sep_line_color and (sep_line_color[3] < 1.0) and ((col_sep_line > 0) or (row_sep_line > 0)):
        canvas = _draw_separation_lines(canvas, sep_line_color, image_area_offset_y, rows, row_spacing, text_adjusted_font_size, max_height_per_row, image_area_offset_x, cols, col_spacing, max_width_per_col, cell_data, col_sep_line, row_sep_line)

    # Draw the content of each cell
    for cell in cell_data:
        x1, y1 = cell["image_upper_left_x"], cell["image_upper_left_y"]
        x2, y2 = cell["image_lower_right_x"], cell["image_lower_right_y"]
        canvas[y1:y2, x1:x2, :] = cell["image"] # Draw image to canvas

        # Draw cell text
        if cell["cell_text"] is not None:
            text_x = round(cell["cell_center_x"] - cell["text_width"]/2)
            text_y = round(y1 - text_adjusted_font_size)
            canvas = _draw_text(canvas, text_x, text_y, cell["cell_text"], c_image_text_config["color"], text_font)

    # --------------------------------------------------------------------
    # Draw column and row text
    # --------------------------------------------------------------------

    x_min, y_min = image_area_offset_x, image_area_offset_y
    for cell in cell_data:
        x_center, y_center = cell["cell_center_x"], cell["cell_center_y"]
        col, row = cell["col"], cell["row"]

        if (col_text is not None) and (cell["row"] == 0):
            text = col_text[col]
            text_width = dummy_draw.textbbox((0, 0), text, font=col_font)[2]
            x = round(x_center - text_width/2)
            y = round(y_min - col_adjusted_font_size)
            canvas = _draw_text(canvas, x, y, text, c_col_text_config["color"], col_font)

        if (row_text is not None) and (cell["col"] == 0):
            text = row_text[row]
            _, _, text_width, text_height = dummy_draw.textbbox((0, 0), text, font=row_font)
            x = round(x_min - row_adjusted_font_size - text_width)
            y = round(y_center - text_height)
            canvas = _draw_text(canvas, x, y, text, c_row_text_config["color"], row_font)

    # Universal stuff
    if title is not None:
        canvas = image_modifier.draw_image_title(canvas, title, top_padding_color=canvas_background_color)
    if max_output_image_size_wh is not None:
        canvas = image_modifier.resize_universal_output_image(canvas, max_output_image_size_wh)
    if save_image_path is not None:
        utils.save_image_to_disk(save_image_path, canvas)
    if display_image:
        show.show_image(canvas)
    if return_image:
        return canvas


def get_all_possible_aspect_ratios(images, flatten:bool=True):
    # TODO: This function is used by `just_show` and should probably be cleaned up a bit
    def _remove_diagonal_elements(matrix: np.ndarray):
        assert len(matrix.shape) == 2
        n, m = matrix.shape
        assert n > 1

        # Get the indices for the diagonal elements
        diagonal_indices = np.arange(matrix.shape[0])

        # Create a boolean mask to exclude the diagonal elements
        mask = np.ones(matrix.shape, dtype=bool)
        mask[diagonal_indices, diagonal_indices] = False

        # Apply the mask to remove the diagonal elements
        result = matrix[mask].reshape(matrix.shape[0], -1)
        return result

    shapes = np.stack([image.shape for image in images])
    heights, widths, channels = shapes[:, 0], shapes[:, 1], shapes[:, 2]

    # Aspect ratio
    aspect_ratios = widths / heights
    ar_diffs = np.abs(aspect_ratios[:, np.newaxis] - aspect_ratios)  # abs(AR_n - AR_m) for n, m in [1, ..., len(aspect_ratios)]
    ar_diffs = _remove_diagonal_elements(ar_diffs)

    if flatten:
        ar_diffs = ar_diffs.flatten()
    return ar_diffs


def _get_grid_parameters(num_images, height, width, max_width=1920, max_height=1080, desired_ratio=9/17, ratio_weight=1.0, scale_weight=1.0, empty_weight=1.0):
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

    @param max_height:
    @param max_width:
    @param desired_ratio:
    @return: cols, rows, scaling_factor, loss_info
    """

    assert num_images > 1
    assert (height >= global_config.image_min_height) and (width >= global_config.image_min_width)

    N = num_images
    h, w = height, width
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
                    ratio_loss = ratio_weight * ratio_loss
                    scale_loss = (1 - a) ** 2 # (2)
                    scale_loss = scale_weight * scale_loss
                    empty_cell_loss = x*y/N - 1 # (3)
                    empty_cell_loss = empty_weight * empty_cell_loss
                    losses[(x, y, a)] = ratio_loss + scale_loss + empty_cell_loss
                    losses_split.append([ratio_loss, scale_loss, empty_cell_loss])

    if len(losses_split) == 0:
        raise RuntimeError("Failed to produce grid parameters. There can be several reasons for this, "
                           "but the most likely cause is having too many/large images")

    # TODO: Should this still be here? It was useful while debugging, but would this ever be used again?
    # if global_config.in_debug_mode:
    #     import pandas as pd
    #     df = pd.DataFrame([[a, b, c, d] for (a, b, c), d in losses.items()], columns=["y", "x", "a", "total"])
    #     df[["ratio_loss", "scale_loss", "empty_loss"]] = np.array(losses_split)
    #     global_config.df = df


    # pick parameters with the lowest loss
    rl, sl, ecl = losses_split[np.argmin(list(losses.values()))]
    loss_info = {"ratio":rl, "scale":sl, "empty_cell":ecl, "total":rl+sl+ecl}
    cols, rows, scaling_factor = min(losses, key=losses.get)
    return cols, rows, scaling_factor, loss_info


def show_grid(
    image_source:ImageSource,
    allow_auto_resize:bool=True,
    title: Optional[str] = None,
    display_image: bool = True,
    return_image: bool = False,
    max_output_image_size_wh: Optional[tuple] = (1920, 1080),
    save_image_path: Optional[str] = None,
    resize_factor: float = 1.0,
    BGR2RGB: bool = False,
    parse_image: bool = True,
) -> Optional[ndarray]:
    """
    Display `image_source` in an appropriate way, regardless of its format.

    :param image_source: An image or list of images. Can handle most image sources: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param allow_auto_resize: If set to True, the function can automatically resize images to achieve a more optimal layout.
           Note that the interaction between `allow_auto_resize=True` and `resize_factor != 1.0` can be somewhat unpredictable.
    :param title: Custom title for the displayed frames. If not provided, the video file name will be used as the title.
    :param display_image: If True, display the processed image using cv2.imshow() or PIL if in jupyter.
    :param return_image: If True, return the processed image as a np.ndarray (H, W, 3).
    :param max_output_image_size_wh: Maximum width and height of the displayed/returned image. If None, nothing will be done.
    :param save_image_path: If provided, save the processed image at the given file path. Default is None (no saving).
    :param resize_factor: Resize factor for each image/frame. Default is 1.0 (no resizing).
    :param BGR2RGB: If True, convert the processed image from BGR to RGB before display/return.
    :param parse_image: If True, will thoroughly check image_source and ensure it's in the correct format

    """
    # --------------------------------------------------------------------
    # # Universal stuff
    # --------------------------------------------------------------------

    checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image)
    if parse_image:
        image_source = parsers.parse_arbitrary_image_source(image_source, resize_factor, BGR2RGB)
    else:
        image_source = parsers.quick_parse_image_source(image_source)
    images = image_source  # A bit cumbersome to carry around, but I wanted to keep `image_source` for the sake of consistency between show-functions

    # If there's only one image just pass it to _ju
    assert len(images) >= 1, f"Expected at least one image, but received `{len(images)=}`"
    if len(images) == 1:
        return show.show_image(image_source=images[0], title=title, display_image=display_image, return_image=return_image,
                               max_output_image_size_wh=max_output_image_size_wh, save_image_path=save_image_path,
                               resize_factor=resize_factor, BGR2RGB=BGR2RGB, parse_image=False)

    # ------------------------------------------
    # Setup
    # ------------------------------------------

    # Image shape
    shapes = np.stack([image.shape for image in images])
    heights, widths, channels = shapes[:, 0], shapes[:, 1], shapes[:, 2]

    # Aspect ratio
    # If the difference in aspect ratios are less than 10%: resize all images to a single shared dimensions
    if max(get_all_possible_aspect_ratios(images)) < 0.1:
        resize_height, resize_width, _ = list(sorted(shapes, key=lambda x: x[1]))[len(widths)//2] # Pick shape from the image with the median width
        for i, image in enumerate(images):
            images[i] = cv2.resize(image, (resize_width, resize_height))

        # recalculate shapes
        shapes = np.stack([image.shape for image in images])
        heights, widths, channels = shapes[:, 0], shapes[:, 1], shapes[:, 2]

    assert np.all(shapes[:, 2] == 3), "All images should be RGB at this point through `parse_image_as_uint8_rgb_numpy_array(images)`"

    # ------------------------------------------
    # Find a suitable layout
    # ------------------------------------------
    height_heuristic, width_heuristic, plot_settings = None, None, "brute_force"

    # (1) All images are of identical shape
    if np.all(shapes[0, :] == shapes[1:, :]):
        plot_settings = "shape_size"
        height_heuristic, width_heuristic = shapes[0][0], shapes[0][1]

    # (2) If all images are wider than tall: resize to a shared width
    elif np.all(widths > heights):
        plot_settings = "width_shared"
        shared_width = int(np.median(widths))
        for i, image in enumerate(images):
            images[i] = image_modifier.resize_image_respect_aspect_ratio(image, shared_width, None)[0]

    # (3) If all images are taller than wide: resize to a shared height
    elif np.all(heights > widths):
        plot_settings = "height_shared"
        shared_height = int(np.median(heights))
        for i, image in enumerate(images):
            images[i] = image_modifier.resize_image_respect_aspect_ratio(image, None, shared_height)[0]

    # Calculate the final width and height heuristic.
    if (height_heuristic is None) and (width_heuristic is None):
        shapes = np.stack([image.shape for image in images])
        height_heuristic, width_heuristic = int(np.mean(shapes[:, 0])), int(np.mean(shapes[:, 1]))

    # ------------------------------------------
    # Create grid image
    # ------------------------------------------

    # Sanity check
    if (height_heuristic is None) or (width_heuristic is None):
        raise RuntimeError("This shouldn't be possible")

    max_width, max_height = 1920, 1080
    if max_output_image_size_wh is not None:
        max_width, max_height = max_output_image_size_wh

    # Get layout
    cols, rows, scaling_factor, loss_info = _get_grid_parameters(
        len(images), height=height_heuristic, width=width_heuristic, max_width=max_width, max_height=max_height, **global_config.config_auto_grid_layout
    )
    if (scaling_factor != 1.0) and allow_auto_resize:
        for i, image in enumerate(images):
            h, w, _ = image.shape
            images[i] = cv2.resize(image, (int(w * scaling_factor), int(h * scaling_factor)))

    # Show grid according to `plot_settings`
    shared_settings = {"max_output_image_size_wh":max_output_image_size_wh, "display_image":False, "c_drop_down_shadow":None,
                       "image_source":images, "rows":rows, "cols":cols, "canvas_background_color":(245,245,245), "return_image":True}
    if plot_settings == "shape_size":
        final = show_grid_configurable(**shared_settings, c_image_spacing=None, c_image_area_margins=None, c_image_resizing=None)
    elif plot_settings in ["width_shared", "height_shared"]:
        image_spacing = {"col_spacing":5, "row_spacing":5, "col_sep_line":1, "row_sep_line":1, "sep_line_color":(175, 175, 175, 0.3)}
        final = show_grid_configurable(**shared_settings, c_image_spacing=image_spacing, c_image_area_margins=None, c_image_resizing=None, c_image_border=None)
    elif plot_settings == "brute_force":
        final = show_grid_configurable(**shared_settings)
    else:
        raise RuntimeError("This shouldn't be possible")

    # ------------------------------------------
    # Wrap up
    # ------------------------------------------

    if title is not None:
        final = image_modifier.draw_image_title(final, title, top_padding_color=shared_settings["canvas_background_color"])
    if save_image_path is not None:
        utils.save_image_to_disk(save_image_path, final)
    if display_image:
        show.show_image(final)
    if return_image:
        return final

__all__ = ["show_grid_configurable", "show_grid"]