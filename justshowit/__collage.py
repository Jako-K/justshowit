from __future__ import annotations
import numpy as np

from __config import *
import __show
import __utils
import __checker
import __parser
import __image_modifier
RectType = List[int]

def _calculate_overlap(rect1:RectType, rect2:RectType) -> Tuple[int, float, float]:
    """
    >> calculate_overlap([0, 50, 100, 150], [50, 0, 150, 100])
    """
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    if (x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3):
        return 0, 0.0, 0.0  # No overlap

    # Calculate the overlapping region
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    # Calculate the number of overlapping pixels for each rectangle
    overlap_rect1 = (x_right - x_left) * (y_bottom - y_top)
    overlap_rect2 = overlap_rect1

    # Calculate the percentage of overlap for each rectangle
    area_rect1 = (x2 - x1) * (y2 - y1)
    overlap_percentage_rect1 = overlap_rect1 / area_rect1

    area_rect2 = (x4 - x3) * (y4 - y3)
    overlap_percentage_rect2 = overlap_rect2 / area_rect2

    assert overlap_rect1 == overlap_rect2, "The intersection should be the same regardless of rectangle"
    return overlap_rect1, overlap_percentage_rect1, overlap_percentage_rect2


def _simple_intersect_check(rect1:RectType, rect2:RectType) -> bool:
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    if (x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3):
        return False  # No overlap
    return True  # Overlap


def _intersects(rect1:RectType, rect2:RectType, overlap_tolerance_percentage_01:float=0.0, overlap_tolerance_pixels:int=np.inf) -> bool:
    # Simple check (cheaper)
    if (overlap_tolerance_percentage_01 == 0.0) and (overlap_tolerance_pixels == np.inf):
        return _simple_intersect_check(rect1, rect2)

    # Detailed check
    intersection_pixels, overlap_rect1_percentage, overlap_rect2_percentage = _calculate_overlap(rect1, rect2)
    if intersection_pixels > overlap_tolerance_pixels:
        return True
    elif max(overlap_rect1_percentage, overlap_rect2_percentage) > overlap_tolerance_percentage_01:
        return True
    return False


def _get_packed(rectangles:Tuple[Tuple[int, int]], canvas_width:int, canvas_height:int, overlap_tolerance_percentage_01:float, collage_sort:str, add_extra_corner:bool):
    """
    # TODO: Write docstring
    # TODO Update drawio: `add_extra_corner` can add an extra corner
    :return:

    Algorithm is explained in `../doc/packing_algo.png`
    """

    # -------------------------
    # Checks
    # -------------------------
    __checker.assert_types([rectangles, canvas_width, canvas_height], [tuple, int, int], ["rectangles", "canvas_width", "canvas_height"])
    if len(rectangles) < 1:
        raise ValueError(f"Expected to find at least one rectangle, but received `{len(rectangles)=}`")
    if len(rectangles) > 1_000:
        warning_message = f"Received a suspiciously large number of rectangles: `{len(rectangles)=}`"
        warnings.warn(warning_message)
    extra_rectangles_message = "`rectangles` is expected to contain tuples of two positive integers"
    if any([(len(r) != 2) for r in rectangles]):
        raise ValueError(f"One or more entries in `rectangles` has a length different then 2. \n{extra_rectangles_message}")
    if any([( (type(w)!=type(h)!=int) or w < 1 or h < 1 ) for (w, h) in rectangles]):
        raise ValueError(f"One or more widths/heights in `canvas_domains` contains an unexpected value. \n{extra_rectangles_message}")

    # -------------------------
    # Packing
    # -------------------------

    # Setup
    rectangles = [(w, h, i) for i, (w, h) in enumerate(rectangles)]
    solutions = []
    if collage_sort == "as_is":
        pass
    elif collage_sort == "area_width_height":
        rectangles = sorted(rectangles, key=lambda r: (r[0]*r[1], r[0], r[1]), reverse=True) # Sort by area, then width, then height
    elif collage_sort == "area_height_width":
        rectangles = sorted(rectangles, key=lambda r: (r[0]*r[1], r[1], r[0]), reverse=True)
    elif collage_sort == "width_height_area":
        rectangles = sorted(rectangles, key=lambda r: (r[0], r[1], r[0]*r[1]), reverse=True)
    elif collage_sort == "height_width_area":
        rectangles = sorted(rectangles, key=lambda r: (r[1], r[0], r[0]*r[1]), reverse=True)
    else:
        raise RuntimeError("Shouldn't have gotten this far")

    corners = [(0, 0)]
    fitness_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Attempt to pack `images` into a canvas of shape (canvas_width, canvas_height)
    for (rect_width, rect_height, rect_id) in rectangles:
        for i, (x_corner, y_corner) in enumerate(corners):
            x2 = x_corner + rect_width
            y2 = y_corner + rect_height
            this_box = [x_corner, y_corner, x2, y2]
            no_intersection = not any([_intersects(this_box, other_box[:4], overlap_tolerance_percentage_01) for other_box in solutions])
            if (0 < x2 < canvas_width) and (0 < y2 < canvas_height) and no_intersection:
                corners.pop(i)
                solutions.append([x_corner, y_corner, x2, y2, rect_id])
                corners.append((x2, y_corner))  # Upper right corner
                corners.append((x_corner, y2))  # Lower left corner
                if add_extra_corner and (y_corner > 0) and (fitness_mask[y_corner-1, x2] != 1): # Not on image border / other rectangles border
                    strip = fitness_mask.copy()[:y_corner, (x2-1)] # Imagine you send a ray from the upper left corner of the just place rectangle. This finds the y of the first intersection.
                    hit = strip[::-1].argmax()
                    y2_new = (y_corner - hit) if (hit != 0) else 0
                    corners.append((x2, y2_new))  # Corner right above "Upper right corner"

                fitness_mask[y_corner:y2, x_corner:x2] = 1
                break
        corners = sorted(corners, key=lambda c: (c[0], c[1]))  # Sort corners by width then by height # TODO add a "if can potentially fill a hole" --> First

    if len(solutions) != len(rectangles):
        return None

    # Fitness
    solutions = np.array(solutions)
    max_x, max_y = solutions[:, 2:4].max(axis=0)
    fitness = fitness_mask[:max_y, :max_x].mean()

    assert len(np.unique(solutions[:, -1])) == solutions.shape[0] == len(rectangles)
    return {"fitness":fitness, "max_x":max_x, "max_y":max_y, "solutions":solutions}


def show_collage(
        image_source: ImageSource,
        canvas_domains:Tuple[Tuple[int, int]]=None,
        crop_smallest_fit:bool=True,
        overlap_tolerance:float=0.1,
        title: Optional[str] = None,
        display_image: bool = True,
        return_image: bool = False,
        max_output_image_size_wh: tuple = None,
        save_image_path: Optional[str] = None,
        resize_factor: float = 1.0,
        BGR2RGB: bool = False,
        parse_image: bool = True
) -> Optional[ndarray]:
    """
    "Pack and display all images from `image_source` into the smallest canvas available from `canvas_domains`."

    :param image_source: An image or list of images. Can handle most image sources: ndarray, url, path, torch, PIL, video_path. For a full description see: TODO insert link
    :param canvas_domains: A tuple of (width, height)-pairs serving as canvases to fit the images. The smallest canvas that can contain all images will be used.
                           If None: canvas_domains = tuple((int(1920 * 10 / 1.25 ** i), int(1080 * 10 / 1.25 ** i)) for i in range(16, 0, -1))
    :param crop_smallest_fit: If True, will crop the final image to avoid unused space. This means the shape of the final image can differ from those specified in canvas_domains
    :param overlap_tolerance: The maximum overlap two images may have - percentage 0-1. Note if one image overlaps 40% while the other only 10%, 40% will be considered.
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

    # --------------------------------------------------------------------
    # Universal stuff
    # --------------------------------------------------------------------

    __checker.assert_valid_universal_inputs(title, display_image, return_image, max_output_image_size_wh, save_image_path, resize_factor, BGR2RGB, parse_image)
    if parse_image:
        image_source = __parser.parse_arbitrary_image_source(image_source, resize_factor, BGR2RGB)
    else:
        image_source = __parser.quick_parse_image_source(image_source)

    # Function specific
    __checker.assert_types(
        [canvas_domains, crop_smallest_fit, overlap_tolerance],
        [tuple, bool, float],
        ["canvas_domains", "crop_smallest_fit", "overlap_tolerance"],
        [1,0,0]
    )

    # `canvas_domains`
    if canvas_domains is None:
        canvas_domains = tuple((int(1920 * 10 / 1.25 ** i), int(1080 * 10 / 1.25 ** i)) for i in range(16, 0, -1))
    if len(canvas_domains) < 1:
        raise ValueError(f"Expected to find at least one canvas domain, but received `{len(canvas_domains)=}`")
    if len(canvas_domains) > 1_000:
        warning_message = f"Received a suspiciously large number of canvas domains: `{len(canvas_domains)=}`"
        warnings.warn(warning_message)
    extra_canvas_message = "`canvas_domains` is expected to contain tuples of two positive integers"
    if any([(len(cd) != 2) for cd in canvas_domains]):
        raise ValueError(f"One or more entries in `canvas_domains` has a length different then 2. \n{extra_canvas_message}")
    if any([ ( (type(w)!=type(h)!=int) or w < 1 or h < 1 ) for (w, h) in canvas_domains]):
        raise ValueError(f"One or more widths/heights in `canvas_domains` contains an unexpected value. \n{extra_canvas_message}")

    max_canvas_width = max([w for (w, _) in canvas_domains])
    max_image_width = max([image.shape[1] for image in image_source])
    if max_canvas_width < max_image_width:
        raise RuntimeError(f"`{max_canvas_width=} < {max_image_width=}`")

    max_canvas_height = max([h for (_, h) in canvas_domains])
    max_image_height = max([image.shape[0] for image in image_source])
    if max_canvas_height < max_image_height:
        raise RuntimeError(f"`{max_canvas_height=} < {max_image_height=}`")

    # `overlap_tolerance`
    __checker.assert_positive_float(overlap_tolerance, zero_allowed=True, max_value_allowed=0.99, variable_name="overlap_tolerance")

    # -------------------------
    # Packing
    # -------------------------

    # Setup
    rectangles = tuple((int(s.shape[1]), int(s.shape[0])) for s in image_source)
    canvas_image = None
    best_fitness, best_packing_solution = -1.0, None

    # Attempt to pack all images into the smallest of the predefined width-height combinations
    for (canvas_width, canvas_height) in canvas_domains:
        if (canvas_width < max_image_width) or (canvas_height < max_image_height): # Can safely be skipped because it will never fit
            continue

        for sorting_algorithm in ["as_is", "area_width_height", "area_height_width", "width_height_area", "height_width_area"]: # ["as_is"]:
            for add_extra_corner in [False, True]:
                results = _get_packed(rectangles, canvas_width, canvas_height, overlap_tolerance, sorting_algorithm, add_extra_corner)

                if (results is not None) and (results["fitness"] > best_fitness): # No solution found
                    best_fitness = results["fitness"]
                    best_packing_solution = results
                    best_packing_solution["sorting_algorithm"] = sorting_algorithm
                    best_packing_solution["add_extra_corner"] = add_extra_corner
                    best_packing_solution["canvas_shape"] = (canvas_width, canvas_height)

        # No solution was found, try with a bigger canvas
        if best_packing_solution is None:
            continue

        # Create a grey canvas according to the specified dimensions
        canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) + 65  # 65 seems to be a pretty versatile grey color i.e. it looks decent no matter the pictures

        if global_config.in_debug_mode:
            [print(f"{k}: {v}") for k,v in best_packing_solution.items() if (k != "solutions")]

        # (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        for (x1, y1, x2, y2, rid) in best_packing_solution["solutions"]:
            image = image_source[rid]
            assert image.shape[:-1] == (y2 - y1, x2 - x1)
            canvas_image[y1:y2, x1:x2, :] = image
        break


    # -------------------------
    # Wrapping up
    # -------------------------
    if canvas_image is None:
        raise RuntimeError("Failed to produce a mosaic image. There can be several reasons for this:\n"
                           "(1) Too large images\n"
                           "(2) Too many images\n"
                           "(3) The widths and heights in `canvas_domains` may be poorly chosen")
    if (best_packing_solution["max_x"] == -1) or (best_packing_solution["max_y"] == -1):
        raise RuntimeError("This should not be possible.")

    if crop_smallest_fit:
        canvas_image = canvas_image[:best_packing_solution["max_y"], :best_packing_solution["max_x"], :]

    # Wrap up
    if title:
        canvas_image = __image_modifier.draw_image_title(canvas_image, text=title)
    if max_output_image_size_wh is not None:
        canvas_image = __image_modifier.resize_universal_output_image(canvas_image, max_output_image_size_wh=max_output_image_size_wh)
    if display_image:
        __show.show_image(canvas_image)
    if save_image_path is not None:
        __utils.save_image_to_disk(save_image_path, canvas_image)
    if return_image:
        return canvas_image

__all__ = ["show_collage"]