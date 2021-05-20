from Eye_model_function_curve import *
from additional_input import *
import scipy.io as sp
from numpy import genfromtxt

#========= INPUT ==============


ratio = 1### how big is one axis compared to the other.
minor_p = 0.075  #minor axis of the dot   (cm)
major_p = minor_p * ratio #major axis of the dot  (cm)
dist_to_center =  [0.9]   # distance from the fish's head to the center of the stimulus (cm) 
angle_to_fish = [-50] #angle between Oy and the line connecting the closet point of the ellipse to the fish
## start from Ox and go to the 
angle_ver = [40]  # angle from Oy to the horizontal of the ellipse (measured by the vertical axis, i.e. major axis, main axis )  
location = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image'
labels = "constant_ver_1" ##labels of stimulus. Eg: vertical ellipse or horizontal ellipse
#==============================


# =============================================================================
# Input: distance (to border), orientation,  angle to fish, major and minor axis of the projected stimulus, location to save data file
# Output: all the properties (minor axis, major axis, aspect ratio, area, position)
# 
# THE PROPERTIES WILL BE CALCULATED IN THIS ORDER: 
#    for angle_fish in angle_to_fish:
#            for dist in dist_to_border:
#                for i in range(len(angle_hor)):
# 
# 
# =============================================================================

normalize_list2 = [minor_p, major_p]
minor_p, major_p = [x/arena_radius for x in normalize_list2]
dist_to_center = [y/arena_radius for y in dist_to_center]
eye = image_computation()
eye.save_eye_properties(labels, dist_to_center, angle_ver, angle_to_fish, minor_p, major_p, location) #### real computation happens here.


