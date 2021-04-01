
# =============================================================================
# THIS BELOW PART DOES NOT NEED TO BE CHANGED ! 
# =============================================================================
import numpy as np
figure_size = 500
arena_radius = (8.7/2)/figure_size# cm
minor_p = 0.065/arena_radius #minor axis of the dot
major_p = 0.065*6/arena_radius #major axis of the dot
distance_xy_right = 0.9  # in cm
distance_xy_left = 0.9  # in cm
sizes = 0.065/arena_radius
dist_to_border =  [1.54/arena_radius, 0.84/arena_radius, 0.42/arena_radius]#distance_xy_right / arena_radius - sizes
dist_to_fish = dist_to_border#[300,350,400,450, 500] #distance from the first dot to the fish
dist_to_plane = 250 #distance from the vertical (spherical) plane to the fish
dist_btw_eye = 0.12/arena_radius # distance between the two eyes
dist_btw_pnt = 100 #distance between the two consecutive dots
body_len = 200 #the length of the fish's body
y_leye = 0 # y position of the left eye 
x_leye = -0.03/arena_radius# x position of the left eye
eye_rad = 0.02/arena_radius # fish eye's radius
num_points = 1 #number of dots we want to show to the fish
h = 0.35/arena_radius # the z position of the fish from the ground
#h = [100, 200, 400, 600, 800]
x_reye = x_leye + dist_btw_eye #x position of  the right eye
y_reye = y_leye #y position of the right eye
z_reye = z_leye = h #the z position of the fish from the ground
angle_retina = 26.5 #angle made btw the normal vector of retina and the horizontal
retina_field = 163


vector_retina_right_x = np.array([np.cos(np.radians(angle_retina)), np.sin(np.radians(angle_retina)),0]) # normal vector of the right retina plane (x,y,z)
vector_retina_right_y = np.array([-np.sin(np.radians(angle_retina)), np.cos(np.radians(angle_retina)),0])
vector_retina_left_x = np.array([-np.cos(np.radians(angle_retina)), np.sin(np.radians(angle_retina)),0]) #normal vector of the left retina plane (x,y,z)
vector_retina_left_y = np.array([np.sin(np.radians(angle_retina)), np.cos(np.radians(angle_retina)),0])
x_center_retina_right = x_reye - eye_rad*np.cos(np.radians(angle_retina))
y_center_retina_right = y_reye - eye_rad*np.sin(np.radians(angle_retina))
x_center_retina_left = x_leye + eye_rad*np.cos(np.radians(angle_retina))
y_center_retina_left = y_center_retina_right
list_of_input = [x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right_x, vector_retina_left_x]
#change coordinate to the coordinate attached to the retina of each eye:
trans_matrix_right_eye = np.array([vector_retina_right_x, vector_retina_right_y, [0,0,1]])
trans_matrix_left_eye  = np.array([vector_retina_left_x, vector_retina_left_y, [0,0,1]])

[x_new_reye, y_new_reye, z_new_reye]  = np.dot(trans_matrix_right_eye,  (np.array([[x_reye],[y_reye],[z_reye]])- np.array([[x_center_retina_right], [y_center_retina_right], [h]])))
[x_new_leye, y_new_leye, z_new_leye]  = np.dot(trans_matrix_left_eye,  (np.array([[x_leye],[y_leye],[z_leye]])- np.array([[x_center_retina_left], [y_center_retina_left], [h]])))

def wiring_eye(x_new_reye, y_new_reye, z_new_reye, x_new_leye, y_new_leye, z_new_leye, eye_rad, retina_field):
        u, v = np.mgrid[np.radians(retina_field/2):np.radians(360 - retina_field/2):1000j, 0:np.pi:1000j]
        ul, vl = np.mgrid[np.radians(retina_field/2):np.radians(360 - retina_field/2):1000j, 0:np.pi:1000j]
        x = x_new_reye + eye_rad*np.cos(u)*np.sin(v)
        y = y_new_reye + eye_rad*np.sin(u)*np.sin(v)
        z = z_new_reye + eye_rad*np.cos(v)
        xl = x_new_leye + eye_rad*np.cos(ul)*np.sin(vl)
        yl= y_new_leye + eye_rad*np.sin(ul)*np.sin(vl)
        zl = z_new_leye + eye_rad*np.cos(vl)
        right_eye_wire = [x,y,z]
        left_eye_wire = [xl,yl,zl]
        return right_eye_wire, left_eye_wire
    
right_eye_wire, left_eye_wire = wiring_eye(x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, eye_rad, retina_field)
eye_list = [x_center_retina_right, y_center_retina_right, right_eye_wire, left_eye_wire, x_leye, y_leye, x_reye, y_reye, h]

