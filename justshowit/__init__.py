# TODO Remove this when debugging is done:
# TODO It's only here so i can import `justshowit` in jupyter notebook while also debugging in pycharm without having to change import statements to `from . _`
import sys as __sys
import os as __os
__sys.path.append(__os.path.abspath(__os.path.dirname(__file__)))

from .config import global_config
from .show import show
from .collage import show_collage
from .parsers import parse_torch_image_batch_as_uint8_rgb_numpy_array, parse_numpy_image_batch_as_uint8_rgb_numpy_array, parse_image_as_uint8_rgb_numpy_array, parse_arbitrary_image_source
from .grid import show_grid_configurable, show_grid
from .video import play_video, parse_video_to_images, parse_video_to_images_fixed_count, show_video
from .image_modifier import draw_text_cv2, draw_text_pillow, draw_image_title
from .utils import get_cv2_video_capture


###################################################################
# TODO - All around
###################################################################

# 20.) Add the possibility to save videos and individual frames with e.g. enter in `VideoPlayer`
# 21.) List torch as an optional package requirement
# 23.) `resize_universal_output_image` can be off by one pixel, fix it. Must be a simple rounded error somewhere
# 25.) Is it a good idea to have \n in errors? I'm thinking that it may fuck up the layout completely if used on a screen/terminal that is not standard size?
# 28.) Check all <NEW_VALUE> prints are correct had a very confusion problem with global_config.min_width and global_config.image_min_width
# 29.) Go through all the places i comment on comment names and check they are still valid
# 30.) Implement a random rotation option to show_collage
# 31.) Implement `play_audio`
# 32.) Implement `play_gif`
# 33.) add checks that the resized image in `resize_image_respect_aspect_ratio` is not below the minimum value defined in the global_config
# 34.) add resize to play_video perhaps even interactively
# 35.) the extra white space on the left handside of the text between these two: `row_text=["A"*100, "B", "C"], row_text=["A"*20, "B", "C"]` are not the same. They should be.
# 36.) Give a warning for `row_text=["A"*10000, "B", "C"]`
# 37.) setup(..., python_requires='>=3.??') what version should I set as a minimum for python, and should there be a minimum version for e.g. numpy as well?
# 38.) Change name on pip
# 39.) `max_output_image_size_wh` produce an image larger than the allowed in `show_collage`
# 40.) Should this be here, it gets triggered super often: "cv2's frame count is unreliable, this may be an indication that cv2 is struggling to read `video_path='/kaggle/input/video-file-for-lane-detection-project/test_video.mp4"
# 41.) Make a show_table function that can nicely display a pandas dataframe or a dict
# 42.) Make a no_parse option to play_video
# 43.) Make a get video info instead of having to use `return_video_details`
# 44.) This is not neccesary: return _parse_numpy(image) in `_parse_path` and it slows the program down by a lot
# 45.) Either cast a tuple to a list or throw and error when you recieve a tuple. Encountered some esoteric errors while testing tuples of image_sources
# 46.) I add to drop using the `from <module> import *` within the init function because the submodules wouldn't be visible to certain IDEs. It would be nice with a check that could ensure all functions, etc. is imported when e.g. something new is added
# 47.) Make a dedicated `show_torch_batch` function, which can reliably be used to plot batch_images

###################################################################
# #TODO Bugs
###################################################################

# Find out why this cannot be parsed correctly
# >> __image.just_show_image(np.ones((100, 200, 3), dtype="uint8") * (255, )*3)
# If the user provide a folder path that does not exists: throw an error e.g.
# >>> show(image, save_image_path="./asdjklæasdasdljks/IMAGE.png")

###################################################################
# TODO - Complete Rewrite
###################################################################
#
# 2.) checker.py                                          DONE
# 3.) collage.py                                          DONE
# 4.) parsers.py                                           DONE
# 5.) grid.py                                             DONE-ish (need tons more testing)
# 6.) video.py                                            DONE
# 7.) __image_modifier                                      DONE
# 8.) __show                                                DONE
