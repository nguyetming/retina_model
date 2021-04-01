import sys
import numpy as np
import os
import pylab as pl

#    import matplotlib.pyplot as mt
sys.path.append(r"C:\Users\Bob\engert_lab\free_swimming_4fish_setup\modules")

from shared import Shared
from Eye_model_function_curve_214 import *

# dddddd dddd
# we need to define the speed profile of the moving dot, and sample speeds
# from it to use for stimulus presentations.

# lets use sigmoid up and sigmid down - i.e. symmetric distribution
# Fs = 50
# xsig = np.arange(0.7)/Fs
# x0 = 3.5/Fs
# maxV = 4.25  # maximum speed (a bit higher since the sigmid will only approach maxV
# slope = 41  # set the slope of response
# Sup = maxV/(1+np.exp(-slope*(xsig-x0)))
# Sdown = maxV/(1+np.exp(slope*(xsig-x0)))

####VARIABLE ON TIME OF STIMULUS:
dt = 1 / 90
ISI = 2.8  # sec before and after the stim
loom_time = 5  # sec

###VARIABLE ON SIZE OF ARENA AND STIMULUS:
arena_radius = 12.6 / 2  # cm - full size
radius = 1  # radius of the dot
edges = 15  # number of edges of the polygon (dot)
angle_to_fish = 70  # angle to Oy
sizes = 0.075 / arena_radius  # REAL SIZE OF DOT IN RESPECT TO ARENA SIZE

###VARIABLE ON POSITION OF STIMULUS:
distance_xy_right = 0.9  # in cm
distance_xy_left = 0.9  # in cm
dist_to_border = distance_xy_right / arena_radius - sizes

###VARIABLE ON PROPERTIES OF EYE:
dist_btw_eye = 0.1 / arena_radius  # distance between the two eyes
x_leye = -0.05 / arena_radius  # x position of the left eye
y_leye = 0  # y position of the left eye
h_eye = 0.5 / arena_radius  # the z position of the fish from the ground
eye_rad = 0.05 / arena_radius  # fish eye's radius

x_reye = x_leye + dist_btw_eye  # x position of  the right eye
y_reye = y_leye  # y position of the right eye
z_reye = z_leye = h_eye  # the z position of the fish from the ground

vector_retina_right = np.array(
    [np.cos(np.radians(26.5)), -np.sin(np.radians(26.5)), 0])  # normal vector of the right retina plane
vector_retina_left = np.array(
    [-np.cos(np.radians(26.5)), -np.sin(np.radians(26.5)), 0])  # normal vector of the left retina plane
list_of_input = [x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right, vector_retina_left]
properties = [eye_rad, h_eye]

angs = np.linspace(0, 360, edges)
list_vertex = []
list_vertex0 = []

### OTHER VARIABLES
max_dot_speed = 0.006  # [norm distance/dt]
distance_traveled = 0.03  # [norm distance]
total_bout_time = 0.2  # [s]
Slope = 90  # exponential slope

bout_delay_ratio = 3
# now define the sigmoidal growth and decay [including idle time at start and end]
t = np.arange(0, total_bout_time / 2 * (bout_delay_ratio + 1) + dt, dt)
t0_up = (total_bout_time / 2 * bout_delay_ratio + total_bout_time / 4)
t0_down = (total_bout_time / 4)
Sup = max_dot_speed / (1 + np.exp(-Slope * (t - t0_up)))
Sdown = max_dot_speed / (1 + np.exp(Slope * (t - t0_down)))
S = np.concatenate((np.array(Sup), np.array(Sdown[1:len(Sdown)])))
T = np.concatenate((t, (t[1:len(Sdown)] + np.max(t))))
Path = np.cumsum(S)  # total path
alpha = Path / distance_xy_right  # change the path to agnles in radians

full_bout_angle = alpha[-1]  # total angle traveled in a bout

Nbouts = loom_time / (total_bout_time * (bout_delay_ratio + 1))  #


####################


def squeeze_point_along_axis(point, scale_factor):  # squeeze a circle to an ellipse along an axis
    return (point[0], point[1] * scale_factor)


def rotate_point_around_origin(coor, angle):  # transform between two coordinate systems rotated by an angle
    angle = np.radians(angle)
    x, y = coor
    xx = x * np.cos(angle) + y * np.sin(angle)
    yy = -x * np.sin(angle) + y * np.cos(angle)

    return xx, yy


class imageToDot(image_computation):
    ## compute vertices position for a normal circle dot
    for j in range(len(angs)):
        ang0 = angs[j]
        list_vertex.append([np.cos(ang0 * np.pi / 180) * radius, np.sin(ang0 * np.pi / 180) * radius])
    for j in range(len(angs)):
        ang0 = angs[j]
        list_vertex0.append([np.cos(ang0 * np.pi / 180) * 0, np.sin(ang0 * np.pi / 180) * 0])

    def compute_image(self, list_of_input, eye_properties):
        eye_rad, h_eye = eye_properties

        x_right = (sizes + dist_to_border) * np.sin(angle_to_fish * np.pi / 180.)
        y_right = (sizes + dist_to_border) * np.cos(angle_to_fish * np.pi / 180.)

        x_left = (sizes + dist_to_border) * np.sin(-angle_to_fish * np.pi / 180.)
        y_left = (sizes + dist_to_border) * np.cos(-angle_to_fish * np.pi / 180.)

        x_pos = [x_right, x_left]
        y_pos = [y_right, y_left]

        points = {}

        for i in range(2):
            points[i] = [np.array([(vertex[1] * sizes + y_pos[i]) for vertex in list_vertex]),  # y
                         np.array([(vertex[0] * sizes + x_pos[i]) for vertex in list_vertex]),  # x
                         np.zeros((len(list_vertex)))]

        x_new_right, y_new_right, z_new_right, x_new_left, y_new_left, z_new_left = self.find_new_coordinate_list(
            points, 2, list_of_input, eye_rad)

        x_new = [x_new_right[0], x_new_left[1]]
        y_new = [y_new_right[0], y_new_left[1]]
        z_new = [z_new_right[0], z_new_left[1]]
        print(x_new)
        return x_new, y_new, z_new, x_pos, y_pos, points

    def traceToDot(self, list_of_input, eye_properties, list_vertex, sizes):
        x_new, y_new, z_new, x_pos, y_pos, points = self.compute_image(list_of_input, eye_properties)

        x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, _, _ = list_of_input
        x_ = [x_reye, x_leye]
        y_ = [y_reye, y_leye]
        z_ = [z_reye, z_leye]

        eye_rad, h_eye = eye_properties
        stimuli = [[list_vertex0], [list_vertex0]]
        raw_stimuli =  [[list_vertex0], [list_vertex0]]
        phi_delta = 3
        eye_rad_zoom = 10
        dx_dy_twoSides = [[[0]], [[0]]]
        stimuli_amount = 8
        normal_fact = 0
        for i in range(2):  # compute next images for both eyes and trace back to the dot on both eyes

            x_new_ = np.array(x_new[i])
            y_new_ = np.array(y_new[i])
            z_new_ = np.array(z_new[i])

            phi_origin = np.arcsin((z_new_ - z_[i]) / eye_rad)

            theta_origin = np.arccos((x_new_ - x_[i]) / (eye_rad * np.cos(phi_origin)))

            theta_mean = np.median(theta_origin)

            y_max = 0
            x_max = 0
            rr_list = []
            cc_list = []

            for j in range(stimuli_amount):
                # try find the polar (phi) and the azimuth angle (theta) of the image:

                ##for minh: if theta is updated and phi is not --> rotate on a same plane around the fish. if phi is updated while theta_mean is used --> elevate around the fish.
                phi_new = phi_origin - np.radians(phi_delta)
                theta_new = (theta_origin - theta_mean) * np.cos(phi_origin) / np.cos(phi_new) + theta_mean

                # find coordinate of the next image on retina
                z_new2 = z_[i] + eye_rad_zoom * np.sin(phi_new)
                x_new2 = x_[i] + eye_rad_zoom * np.cos(phi_new) * np.cos(theta_new)
                y_new2 = y_[i] - eye_rad_zoom * np.cos(phi_new) * np.sin(theta_new)

                # Trace back to find the real dots and plot them

                dots = self.dot_find(x_new_, y_new_, z_new_, x_[i], y_[i], z_[i], eye_rad)



                x_mean = np.mean(dots[1])
                y_mean = np.mean(dots[0])
                dx_dy_twoSides[i].append([np.sqrt(x_mean ** 2 + y_mean ** 2)])

                rr = [(j - y_mean) for j in dots[0]]  # y
                cc = [(k - x_mean) for k in dots[1]] # x

                if i == 0 and j == 0:
                    normal_fact = max(max(rr), max(cc))


                if len(rr) > 0:
                    if max(rr) > y_max:
                        y_max = max(abs(np.array(rr)))
                    if max(cc) > x_max:
                        x_max = max(abs(np.array(cc)))
                rr_list.append(rr)
                cc_list.append(cc)

                z_new_ = z_new2
                x_new_ = x_new2
                y_new_ = y_new2
                phi_origin = phi_new
                theta_origin = theta_new



                vertices = [[cc_list[j][i] / normal_fact, rr_list[j][i] / normal_fact] for i in range(len(rr_list[j]))]
                stimuli[i].append(vertices)
                raw_vertices = [[dots[1][i] , dots[0][i] ] for i in range(len(dots[0]))]
                raw_stimuli[i].append(raw_vertices)

        print("len stimuli:", len(stimuli))
        return stimuli, dx_dy_twoSides, normal_fact, points, raw_stimuli

    def mix_two_list(self, list1, list2, ISI, loom_time):
        list_mix = []
        t_stimuli = []
        for i in range(len(list1)):
            for j in range(len(list2)):
                if i != j:
                    list_mix.append([list1[i], list2[j]])
                    t_stimuli.append(2 * ISI + loom_time)
        return list_mix, t_stimuli


image_reverse = imageToDot()
stimuli, dx_dy_twoSides, scale, points, raw_stimuli = image_reverse.traceToDot(list_of_input, properties, list_vertex, sizes)
print("len stimuli one eye:", len(stimuli[0]))
print("len stimuli two eye combined:", len(stimuli))
#stimuli, t_stimuli = image_reverse.mix_two_list(stimuli[0], stimuli[1], ISI, loom_time)
#raw_stimuli, _ = image_reverse.mix_two_list(raw_stimuli[0], raw_stimuli[1], ISI, loom_time)
#dx_dy_twoSides, _ = image_reverse.mix_two_list(dx_dy_twoSides[0], dx_dy_twoSides[1], ISI, loom_time)
print("scale:",scale)
print("dot on right side:", points[0])
####Suppose the dot is at x,y from the fish

print(len(stimuli[0]), len(raw_stimuli[0]))
print("trial length", 2 * ISI + loom_time)
print("distance_to_fish:", dx_dy_twoSides)
# stimuli = [0]

#####This is to plot patch to see how the dots look like:



from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
def sort_vertex(vertex_list):
    cent = (sum([p[0] for p in vertex_list]) / len(vertex_list),
            sum([p[1] for p in vertex_list]) / len(vertex_list))
    # sort by polar angle
    vertex_list.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
    return vertex_list

fig, ax0 = plt.subplots(1,1)
patches = []
print(len(stimuli))
for i in range(2):
    vertices_right = raw_stimuli[0][i]
    vertices_right = sort_vertex(vertices_right)
    print("index:", i, vertices_right)
    polygon = mpl.patches.Polygon(np.array(vertices_right))
    patches.append(polygon)
p = PatchCollection(patches, cmap=mpl.cm.jet, alpha=0.04)
colors = 100 * np.random.rand(len(patches))
p.set_array(np.array(colors))
ax0.set_xlim(-1,1)
ax0.set_ylim(-1,1)
ax0.add_collection(p)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# fig1, ax1 = plt.subplots(1,1)
# ax1.scatter(points[0][1], points[0][0])
# plt.show()


x = []
y = []
for i in range(len(raw_stimuli)):
    vertices_right = raw_stimuli[0][i]
for j in range(len(vertices_right)):
    x.append(vertices_right[j][0])
    y.append(vertices_right[j][1])

plt.scatter(x,y)
#plt.show()