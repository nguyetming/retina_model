import numpy as np
figure_size = 2000

#####SAVE ALL THE VARIABLES IN CM ONLY! 


#####===================== ADJUSTABLE VARIABLES ===============================
arena_radius_real = 9.2/2                                     # cm
distance_xy_right = 0.9  # in cm
distance_xy_left = 0.9  # in cm                                          cm
dist_btw_eye = 0.12 # distance between the two eyes        cm
body_len = 0.5 #the length of the fish's body              cm
y_leye = 0 # y position of the left eye                    cm
x_leye = -0.06# x position of the left eye                 cm
eye_rad = 0.045 # fish eye's radius                         cm
h = 0.5 # the z position of the fish from the ground      cm
x_reye = x_leye + dist_btw_eye #x position of  the right eye    cm
y_reye = y_leye #y position of the right eye               cm
z_reye = z_leye = h #the z position of the fish from the ground  cm
angle_retina = 26.5 #angle made btw the normal vector of retina and the horizontal
retina_field = 163
####===========================


num_points = 1
arena_radius = (arena_radius_real)/figure_size
print("arena",arena_radius)
normalize_list = [dist_btw_eye, body_len,  eye_rad, h, x_reye, y_reye, z_reye, x_leye, y_leye, z_leye]
[dist_btw_eye, body_len, eye_rad, h, x_reye, y_reye, z_reye,  x_leye, y_leye, z_leye ]= [x/arena_radius for x in normalize_list]



vector_retina_right_x = np.array([np.cos(np.radians(angle_retina)), np.sin(np.radians(angle_retina)),0]) # normal vector of the right retina plane (x,y,z)
vector_retina_right_y = np.array([-np.sin(np.radians(angle_retina)), np.cos(np.radians(angle_retina)),0])
vector_retina_left_x = np.array([-np.cos(np.radians(angle_retina)), np.sin(np.radians(angle_retina)),0]) #normal vector of the left retina plane (x,y,z)
vector_retina_left_y = np.array([np.sin(np.radians(angle_retina)), np.cos(np.radians(angle_retina)),0])
x_center_retina_right = x_reye - eye_rad*np.cos(np.radians(angle_retina))
y_center_retina_right = y_reye - eye_rad*np.sin(np.radians(angle_retina))
print("y center retina:", y_center_retina_right * arena_radius)
x_center_retina_left = x_leye + eye_rad*np.cos(np.radians(angle_retina))
y_center_retina_left = y_center_retina_right
list_of_input = [x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right_x, vector_retina_left_x]
#change coordinate to the coordinate attached to the retina of each eye:
# trans_matrix_right_eye = np.array([vector_retina_right_x, vector_retina_right_y, [0,0,1]])
# trans_matrix_left_eye  = np.array([vector_retina_left_x, vector_retina_left_y, [0,0,1]])

# [x_new_reye, y_new_reye, z_new_reye]  = np.dot(trans_matrix_right_eye,  (np.array([[x_reye],[y_reye],[z_reye]])- np.array([[x_center_retina_right], [y_center_retina_right], [h]])))
# [x_new_leye, y_new_leye, z_new_leye]  = np.dot(trans_matrix_left_eye,  (np.array([[x_leye],[y_leye],[z_leye]])- np.array([[x_center_retina_left], [y_center_retina_left], [h]])))


def wiring_eye(x_new_reye, y_new_reye, z_new_reye, x_new_leye, y_new_leye, z_new_leye, eye_rad, retina_field):
        u, v = np.mgrid[np.radians(retina_field/2 + 26.5):np.radians(360 - retina_field/2 + 26.5):20j, 0:np.pi:20j]
        ul, vl = np.mgrid[np.radians(retina_field + 72):np.radians(360+72):20j, 0:np.pi:20j]
        x = x_new_reye + eye_rad*np.cos(u)*np.sin(v)
        y = y_new_reye + eye_rad*np.sin(u)*np.sin(v)
        z = z_new_reye + eye_rad*np.cos(v)
        xl = x_new_leye + eye_rad*np.cos(ul)*np.sin(vl)
        yl= y_new_leye + eye_rad*np.sin(ul)*np.sin(vl)
        zl = z_new_leye + eye_rad*np.cos(vl)
        right_eye_wire = [x,y,z]
        left_eye_wire = [xl,yl,zl]
        return right_eye_wire, left_eye_wire
    
right_eye_wire, left_eye_wire = wiring_eye(x_reye * arena_radius, y_reye* arena_radius, z_reye* arena_radius,
                                           x_leye* arena_radius, y_leye* arena_radius, z_leye* arena_radius, eye_rad* arena_radius, retina_field)
eye_list = [x_center_retina_right, y_center_retina_right, right_eye_wire, left_eye_wire, x_leye, y_leye, x_reye, y_reye, h]

