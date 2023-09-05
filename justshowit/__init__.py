# TODO Remove this when debugging is done:
# TODO It's only here so i can import `justshowit` in jupyter notebook while also debugging in pycharm without having to change import statements to `from . _`
import sys as __sys
import os as __os
__sys.path.append(__os.path.abspath(__os.path.dirname(__file__)))

from __config import global_config
from __show import show
from __collage import *
from __parser import *
from __grid import *
from __video import *
from __image_modifier import *

###################################################################
# TODO - All over
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

###################################################################
# #TODO Bugs
###################################################################

# Find out why this cannot be parsed correctly
# >> __image.just_show_image(np.ones((100, 200, 3), dtype="uint8") * (255, )*3)

###################################################################
# TODO - Complete Rewrite
###################################################################
#
# 2.) __checker.py                                          DONE
# 3.) __collage.py                                          DONE
# 4.) __parser.py                                           DONE
# 5.) __grid.py                                             DONE-ish (need tons more testing)
# 6.) __video.py                                            DONE
# 7.) __image_modifier                                      DONE
# 8.) __show                                                DONE
