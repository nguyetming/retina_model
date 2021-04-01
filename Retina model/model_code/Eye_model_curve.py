from Eye_model_function_curve import *
from additional_input import *
from celluloid import Camera
import scipy.io as sp
#Input the data here:
#angle_to_fish = [0,30.,45.,60., 90., 120.,135.,150.]
#angle_hor = [0,30.,45.,60., 90., 120.,135.,150.]

angle_to_fish = [60] #angle between Ox and the line connecting the closet point of the ellipse to the fish
dist_to_border =  [1.54/arena_radius, 0.84/arena_radius, 0.42/arena_radius]  # --> dist to border #distance_xy_right / arena_radius - sizes
angle_hor = [-45,0,45,90]  # angle to horizontal of the ellipse (measured by the vertical axis, i.e. major axis, main axis )

eye = image_computation()
ratio = 6 
minor_p = 0.065/arena_radius #minor axis of the dot
major_p = 0.065* ratio/arena_radius #major axis of the dot


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

location = r'/Users/nguyetnguyen/Documents/RH/eye model/image_data.csv'
labels = ["5135", "590", "545", "50", "10135","1090","1045","100", "45135","4590", "4545", "450"]
eye.save_eye_properties(labels, dist_to_border, angle_hor, angle_to_fish, minor_p, major_p, location)





