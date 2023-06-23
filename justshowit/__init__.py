from .__show_logic import show

###################################################################
# 								TODO
###################################################################

# 1.) Make a show_video - also allow for videos passed as list e.g. np.array of shape (frames, height, width, color_channels)
# 2.) Plot a whole batch
# 3.) Add some sort of warning if a tensor that has been normalized is passed to `show`. Normally their values spand just beyond 0-1 range at this should be detectable in most instances
# 4.) Add shape take for numpy input images 
# 5.) Grid show function. It should be have user defined rows + cols + individual titles + spacing + other quality of life stuff
# 6.) Make the _get_image function public. I think it should be as simple as adding checks?
# 7.) Add a show torch_batch with added renormalized capability.

# 9.) Better custom packing algorithm:
#       a.) Make a heuristic that works by calculating how much of the image could be cropped away if the rect is placed at any given location
#       b.) Make a NN heuristic. It could take some fixed of rects as input e.g. 200 x 4 + canvas_dims. The loss could simply be the less over space after cropping?
#           The model could perhaps be allowed to overlap images, but penalize it relatively hard?
#           Instead of outputting e.g. 200 x 4 it could be 200 x 5, where the extra number that say, hey this image doesn't really fit, sigmoid score?


###################################################################
# 						   TODO - Done
###################################################################

# 1.) Write you own packer. I really don't like that I'm relying on `rectpack`.
#     It's a heavily object-oriented project which, to me at least, is very hard to reason about because it's so steeped in abstractions and indirections.
#     It would be much better to have a self-contained function that do exactly what I need it to and no more. It would also shield me against future issues.
