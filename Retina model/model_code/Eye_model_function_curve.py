#!/usr/bin/env python
# coding: utf-8

# In[61]:


import matplotlib.pyplot as plt


import seaborn as sns
sns.set()
import pandas as pd

import numpy as np
import skimage
from skimage.draw import circle, ellipse,line, polygon
from skimage.util.shape import view_as_windows
from matplotlib.pyplot import figure
from celluloid import Camera
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
import scipy.io as sp
from additional_input import * 
import matplotlib
# In[62]:

#This function is to draw the fish, with x and y of left eye, the body length and the distance between eyes.
class image_computation:
    def draw_fish(self, y_leye, x_leye, eye_rad, dist_btw_eye, body_len, dist_to_fish):
        img = np.zeros((3*np.max(dist_to_fish), 3*np.max(dist_to_fish)), dtype=np.uint8)
        rr_lef, cc_lef = circle(y_leye, x_leye, eye_rad)#ellipse(1000, 1500, 100,200, rotation=np.deg2rad(90)) #circle(500, 500, 60)
        rr_ri, cc_ri = circle(y_leye, x_leye + dist_btw_eye, eye_rad)
        body_y = np.array([y_leye, y_leye, y_leye - body_len])
        body_x = np.array([x_leye, x_leye + dist_btw_eye, x_leye + dist_btw_eye/2])
        rr_tail, cc_tail = polygon(body_y, body_x)
        #body_y, body_x = ellipse(y_leye+ body_len/2, x_leye + dist_btw_eye/2, 30, body_len/2, rotation=np.deg2rad(90))
        img[rr_tail, cc_tail] = 2
        img[rr_lef, cc_lef] = 1
        img[rr_ri, cc_ri] = 1
     
        return img
    
    
    #create a sphere 
    def sphere(self, shape, radius, position):
        # assume shape and position are both a 3-tuple of int or float
        # the units are pixels / voxels (px for short)
        # radius is a int or float in px
        semisizes = (radius,) * 3
    
        # genereate the grid for the support points
        # centered at the position indicated by position
        grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
        position = np.ogrid[grid]
        # calculate the distance of all points from `position` center
        # scaled by the radius
        arr = np.zeros(shape, dtype=float)
        for x_i, semisize in zip(position, semisizes):
            # this can be generalized for exponent != 2
            # in which case `(x_i / semisize)`
            # would become `np.abs(x_i / semisize)`
            arr += (x_i / semisize) ** 2
       
        # the inner part of the sphere will have distance below 1
        return  arr <= 1.0 # ((arr > 0.95) & (arr <= 1.0))
    
    
    #This function is to draw dots that will be shown to the fish, with angle from the first dot to the fish, the angle of the three dots to the horizontal
    def draw_fish_dot(self, num_points, angle_to_fish, angle_hor, y_leye, x_leye, minor_p, major_p, dist_btw_eye, h,  dist_to_fish, dist_to_plane, dist_btw_pnt, direction):
        x_center = x_leye + dist_btw_eye/2
        y_center = y_leye
        if direction == 'horizontal':
            y_dot = y_center + dist_to_fish*np.sin(np.radians(angle_to_fish))
            x_dot = x_center + dist_to_fish*np.cos(np.radians(angle_to_fish))
            points = {}
    	
    	
            for i in range(num_points):
    	  
                rr_p1, cc_p1 = ellipse(y_dot + major_p * np.sin(angle_hor/180*np.pi), x_dot + major_p * np.cos(angle_hor/180*np.pi),minor_p, major_p,  rotation=np.deg2rad(-angle_hor) )
               
                points[i] = rr_p1, cc_p1, np.zeros(len(rr_p1))
                y_dot = y_dot + dist_btw_pnt*np.sin(np.radians(angle_hor))
                x_dot = x_dot + dist_btw_pnt*np.cos(np.radians(angle_hor))
                
    
        if direction == 'vertical':
            #dist_to_fish = distance the dot to the projection image of the fish to the plane.
            #angle_to_fish is now angle to the projection image of the fish to the vertical plane.
            y_dot = y_center + dist_to_plane
            x_dot = x_center + dist_to_fish*np.cos(np.radians(angle_to_fish))
            z_dot = h + dist_to_fish*np.sin(np.radians(angle_to_fish))
            points = {}
            for i in range(num_points):
              rr_p1, cc_p1 = ellipse(z_dot,x_dot,minor_p, major_p,  rotation=np.deg2rad(-angle_hor))
              points[i] = y_dot*np.ones(len(rr_p1)), cc_p1, rr_p1
              z_dot = z_dot + dist_btw_pnt*np.sin(np.radians(angle_hor))
              x_dot = x_dot + dist_btw_pnt*np.cos(np.radians(angle_hor))
    
           
            
        if direction == 'spherical':
            points = {}
            dot = sphere((h+dist_to_plane, 3*major_p, 3*major_p), major_p, (h + round(dist_to_plane*np.sin(np.radians(angle_to_fish))),
                                                                            3/2*major_p,
                                                                            3/2*major_p))
            points[0] = np.where(dot == 1)[1] + (y_center + dist_to_plane*np.cos(np.radians(angle_to_fish))*np.sin(np.radians(angle_hor)) - 3/2*major_p),\
                        np.where(dot==1)[2] + (x_center + dist_to_plane*np.cos(np.radians(angle_to_fish))*np.cos(np.radians(angle_hor)) - 3/2*major_p),\
                        np.where(dot == 1)[0]
    
       
            
        return points
    
    def draw_pattern(self, edge,width_patch):
         start = 0
         black_white = np.zeros([edge,edge])
         start = 0
         for i in range(int(edge/width_patch)):
             
             black_white[:, start:start+width_patch] = np.mod(i+1,2)
             
             if i == int(edge/width_patch) - 1 and start+ width_patch < edge:
                 black_white[:, start + width_patch : edge] = np.mod(i,2)
                 
             start += width_patch
    
         return black_white
    
    
    #This function is to find the coordinate of each point's image on the retina
    def image_find(self, x_p, y_p, z_p, x_eye, y_eye, z_eye, eye_rad):
        #PE is the distance from Point to Eye
        PE = np.sqrt((x_p - x_eye)**2 + (y_p - y_eye)**2 + (z_p - z_eye)**2)
        x = (x_eye - x_p)*eye_rad/PE + x_eye
        y = (y_eye - y_p)*eye_rad/PE + y_eye
        z = (z_eye - z_p)*eye_rad/PE + z_eye
        return x,y,z
    
    
    #base on the number of angle -> approxiamte the image area on retina.
    # In[75]:
    
    #Determine which eye can see the dots shown to the fish
    def angle_to_retina(self, x_p, y_p, z_p, x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right,vector_retina_left):
        vector_preye = np.array([x_p - x_reye ,y_p - y_reye,z_p - z_reye])
        vector_pleye = np.array([x_p - x_leye ,y_p - y_leye,z_p - z_leye])
        
        unit_vector_1r = vector_preye / np.linalg.norm(vector_preye)
        unit_vector_2r = vector_retina_right / np.linalg.norm(vector_retina_right)
        dot_productr = np.dot(unit_vector_1r, unit_vector_2r)
        angle_btw_point_et_retina_right = np.arccos(dot_productr)
    
        unit_vector_1l = vector_pleye / np.linalg.norm(vector_pleye)
        unit_vector_2l = vector_retina_left / np.linalg.norm(vector_retina_left)
        dot_productl = np.dot(unit_vector_1l, unit_vector_2l)
        angle_btw_point_et_retina_left = np.arccos(dot_productl)
        
        return angle_btw_point_et_retina_right, angle_btw_point_et_retina_left
    
    
    # In[80]:
    
    #Find the images of the dots on the retina
    def find_new_coordinate_list(self, points,num_points, list_of_input, eye_rad):
        x_new_right = {}
        y_new_right = {}
        x_new_left = {}
        y_new_left = {}
        z_new_right = {}
        z_new_left = {}
        x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right_x, vector_retina_left_x = list_of_input
        retina_field = 163
        
        for i in range(num_points):
            x_new_right[i] = []
            y_new_right[i] = []
            x_new_left[i]  = []
            y_new_left[i]  = [] 
            z_new_right[i] = []
            z_new_left[i]  = [] 
            for j in range(points[i][0].shape[0]):
                y_p = points[i][0][j]
                x_p = points[i][1][j]
                z_p = points[i][2][j]
                angle_btw_point_et_retina_right, angle_btw_point_et_retina_left = self.angle_to_retina(x_p, y_p, z_p, x_reye, y_reye, z_reye, x_leye, y_leye, z_leye, vector_retina_right_x,vector_retina_left_x)   
                if angle_btw_point_et_retina_right <= np.radians(retina_field/2):
                    
                    x_new,y_new,z_new = self.image_find(x_p, y_p, z_p, x_reye, y_reye, z_reye, eye_rad)
                    x_new_right[i].append(x_new)
                    y_new_right[i].append(y_new)
                    z_new_right[i].append(z_new)
                else: 
                    print("toofar")
                if angle_btw_point_et_retina_left <= np.radians(retina_field/2):
                   
                    x_new,y_new,z_new = self.image_find(x_p, y_p, z_p, x_leye, y_leye, z_leye, eye_rad)
                    x_new_left[i].append(x_new)
                    y_new_left[i].append(y_new)
                    z_new_left[i].append(z_new)
    
        return x_new_right, y_new_right,z_new_right, x_new_left, y_new_left,z_new_left
    
    #Draw expected dot's image on the retina of the fish
    def draw_dot_onretina(self, num_points, angle_hor, radi_dot, x_dot, y_dot, dist_btw_pnt):
        img = np.zeros((400, 400), dtype=np.uint8)
        points = {} 
        for i in range(num_points):
          rr_p1, cc_p1 = circle(x_dot + 200, y_dot + 200,radi_dot)
          img[rr_p1, cc_p1] = 3+i
          points[i] = rr_p1 - 200, cc_p1 - 200
          y_dot = y_dot + dist_btw_pnt*np.sin(np.radians(angle_hor))
          x_dot = x_dot + dist_btw_pnt*np.cos(np.radians(angle_hor))
        #plt.imshow(img,origin='lower')
          
        return img, points
    #Find dots from position of dot's image on retina
    def dot_find(self, x_p, y_p, z_p, x_eye, y_eye, z_eye, eye_rad):
        #PE/eye_rad = (z_eye - z)/(z_p-z_eye)
        PE = (z_eye)/(z_p-z_eye)*eye_rad
        x = (x_eye - x_p)*PE/eye_rad + x_eye
        y = (y_eye - y_p)*PE/eye_rad + y_eye
        z = np.zeros(len(x_p))
        return y, x, z
    
    
    #Compute area and aspect ratio of the image on the retina:
    def compute_area_and_aspect_ratio(self, x_list, y_list, z_list, eye_rad):
        semi_minor = 0
        semi_major = 0
        if x_list != []:
            semi_major = np.sqrt((np.max(x_list)-np.min(x_list))**2 + (np.max(y_list)-np.min(y_list))**2)/2
            #position of points with max and min z:
            x_h_max = x_list[np.where(z_list == np.max(z_list))[0][0]]
            y_h_max = y_list[np.where(z_list == np.max(z_list))[0][0]]
            x_h_min = x_list[np.where(z_list == np.min(z_list))[0][0]]
            y_h_min = y_list[np.where(z_list == np.min(z_list))[0][0]]
            
            semi_minor = np.sqrt((np.max(z_list)-np.min(z_list))**2 + (y_h_max - y_h_min)**2 + (x_h_max - x_h_min)**2)/2
            #find area of the plane ellipse 
            area_perpendicular = np.pi*semi_major*semi_minor
            #find a spherical cap's radius with equivalent area: 
            equivalent_radius = np.sqrt(area_perpendicular/np.pi)
            #find the height of the cap:
            h_plane_to_edge = eye_rad - np.sqrt(eye_rad**2 - equivalent_radius**2)
            #find area of the spherical cap:
            area = 2*np.pi*eye_rad*h_plane_to_edge
            ratio = semi_major/semi_minor
        else:
            area = 0
            ratio = 0
        return area, ratio, semi_major, semi_minor
    
    def compute_position_to_center(self,  x_list, y_list, z_list, eye_list):
        [x_center_retina_right, y_center_retina_right, _, _, _, _,_, _, h] = eye_list
        distance = 0
        if x_list != []:
            
            x_med = np.median(x_list)
            y_med = np.median(y_list)
            z_med = np.median(z_list)
            distance = np.sqrt((x_med - x_center_retina_right)**2 + (y_med - y_center_retina_right)**2 + (z_med -h)**2)
            altitude = np.arcsin((z_med - h)/ distance)/np.pi*180
            vector_image_center = np.array([x_med -  x_center_retina_right, y_med -  y_center_retina_right, 0])
            vector_eye_center = np.array([x_reye -  x_center_retina_right, y_reye -  y_center_retina_right, 0])
            inner_prod = vector_image_center.dot(vector_eye_center)
            print(inner_prod)
            azimuth = np.arccos(inner_prod/ (np.linalg.norm(vector_image_center) * np.linalg.norm(vector_eye_center))) / np.pi * 180
            print('alzimuth:', azimuth)
        return distance, altitude, azimuth
    
    #Create a wiring eye:
        
    
    
    def wiring_eye(self, x_new_reye, y_new_reye, z_new_reye, x_new_leye, y_new_leye, z_new_leye, eye_rad, retina_field):
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
    
    
    # Plot the image on the eye
    def plot_eye_with_image(self, camera, axes, x_list, y_list, z_list, points, num_points, eye_list):
        [x_center_retina_right, y_center_retina_right, right_eye_wire, left_eye_wire, x_leye, y_leye, x_reye, y_reye, h] = eye_list
        [x_right, x_left] = x_list
        [y_right, y_left] = y_list
        [z_right, z_left] = z_list
        [x,y,z] = right_eye_wire
        [xl, yl, zl] = left_eye_wire
        
        [ax0, ax1] = axes #, ax2, ax3] = axes
        colors = ['b', 'r','g','purple']
        for j in range(num_points):
            ax1.scatter(x_right[j], y_right[j], z_right[j], marker='o', color = colors[j])
            ax1.scatter(x_center_retina_right, y_center_retina_right, h, marker='o', s = 200, color = 'red')
            #ax2.scatter(x_left[j], y_left[j], z_left[j], marker='o', color = 'b')
            #ax2.scatter(0,0,0, marker='o', s = 200, color = 'red')
            
            #ax0.set_xlabel('Left ------> Right')
            #ax0.set_ylabel('Back ------> Front')
            #ax0.imshow(img, origin = "lower",extent=[-img.shape[1]/2, img.shape[1]/2, -img.shape[1]/2, img.shape[1]/2])
            ax0.scatter(points[j][1], points[j][0], s = 10, marker='o', color = colors[j])
        ax1.plot_wireframe(x, y, z, color="lightblue")
        #ax1.plot_wireframe(xl, yl, zl, color="lightblue")
        #ax0.scatter(points[0][1], points[0][0], points[0][2], s = 10, marker='o', color = 'b')
        ax0.scatter(x_leye, y_leye, marker='o', s = 60, color = 'r')
        ax0.scatter(x_reye, y_reye, marker='o', s= 60, color = 'r')
        ax0.set_aspect('equal')
        x = [x_reye, x_leye,(x_reye + x_leye)/2 , (x_reye + x_leye)/2]
        y = [y_reye, y_leye, y_reye , y_reye - 200]
        z = [h, h, h - 10, h]
    
        vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    
        tupleList = list(zip(x, y, z))
    
        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        #ax0.add_collection3d(Poly3DCollection(poly3d, facecolors='b', linewidths=1, alpha=0.5))
    
        camera.snap()
        return camera
    
    #Plot the change of area with the change in value of the variable we choose.
    def plot_area_aspect_ratio(self, area, ratio, distance, list_of_value):
      
        fig, axs = plt.subplots(3,1)
        axs[0].plot(list_of_value, area, 'go--', linewidth = 2, markersize = 12)
        
    
        axs[0].set_xlabel('Angle from fish', fontsize= 14 )
        axs[0].set_ylabel('Area of the image \n on the retina (in pixel)', fontsize=13)
        axs[0].tick_params(axis='x', labelsize= 13)
        axs[0].tick_params(axis='y', labelsize= 13)
        
        axs[1].plot(list_of_value, ratio, 'bo--', linewidth = 2, markersize = 12)
        axs[1].set_xlabel('Angle from fish', fontsize=14)
        axs[1].set_ylabel('Aspect ratio of the image \n on the retina', fontsize=13)
        axs[1].tick_params(axis='x', labelsize= 13)
        axs[1].tick_params(axis='y', labelsize= 13)
        
        axs[2].plot(list_of_value, distance, 'ro--', linewidth = 2, markersize = 12)
        axs[2].set_xlabel('Angle from fish', fontsize=14)
        axs[2].set_ylabel('Distance of the image \n to the retina center', fontsize=13)
        axs[2].tick_params(axis='x', labelsize= 13)
        axs[2].tick_params(axis='y', labelsize= 13)
        plt.show()
    
    def save_eye_properties(self, labels, dist_to_border, angle_hor, angle_to_fish, minor_p, major_p, filename):

        semi_major_list = []
        semi_minor_list = []
        area_list = []
        ratio_list = []
        distance_right = []
        altitude_list = []
        azimuth_list = []
        direction = "horizontal"
        u, v = np.mgrid[np.radians(55):np.radians(197+55):1000j, 0:np.pi:1000j]
        x = x_reye + eye_rad * np.cos(u)*np.sin(v)
        y = y_reye + eye_rad * np.sin(u)*np.sin(v)
        z = z_reye+ eye_rad * np.cos(v)
        xl = x_leye + eye_rad * np.cos(u)*np.sin(v)
        yl= y_leye + eye_rad * np.sin(u)*np.sin(v)
        zl = z_leye+ eye_rad * np.cos(v)
        
        
        x_center_retina_right = x_reye - eye_rad*np.cos(np.radians(26.5))
        y_center_retina_right = y_reye + eye_rad*np.sin(np.radians(26.5))
        x_center_retina_left = x_leye + eye_rad*np.cos(np.radians(26.5))
        y_center_retina_left = y_center_retina_right
        
        color = ['r', 'g', 'b']
        
        for angle_fish in angle_to_fish:
            for dist in dist_to_border:
                for i in range(len(angle_hor)):
                    #print("parameters:",num_points, angle_to_fish, angle_hor[i], y_leye, x_leye, minor_p, major_p, dist_btw_eye, h,  dist, dist_to_plane, dist_btw_pnt, direction)
                    points =  self.draw_fish_dot(1, angle_fish, angle_hor[i], y_leye, x_leye, minor_p, major_p, dist_btw_eye, h,  dist, dist_to_plane, dist_btw_pnt, direction)
                    #ax0.scatter(points[0][1], points[0][0], s = 10, marker='o', color = 'purple')
                    
                    #find coordinate of the dots' images on the retina:
                    x_new_right, y_new_right,z_new_right, x_new_left, y_new_left,z_new_left  = self.find_new_coordinate_list(points,num_points, list_of_input, eye_rad)
                    area, ratio, semi_major, semi_minor =   self.compute_area_and_aspect_ratio(x_new_right[0], y_new_right[0], z_new_right[0], eye_rad)  
                    distance_to_center, altitude, azimuth  = self.compute_position_to_center(x_new_right[0], y_new_right[0], z_new_right[0], eye_list)
                    distance_right.append(distance_to_center)
                    altitude_list.append(altitude)
                    azimuth_list.append(azimuth)
                    area_list.append(area)
                    ratio_list.append(ratio)
                    semi_major_list.append(semi_major)
                    semi_minor_list.append(semi_minor)
                    #plot the images on the retina
                    x_list = [ x_new_right, x_new_left]
                    y_list = [ y_new_right, y_new_left]
                    z_list = [ z_new_right, z_new_left]
                
                
        
        #animation = camera.animate(interval=300)
        #plt.show()
            #animation.save('movedotsaway.mp4')
            #plot area and aspect ratio of the images with different distances. 
        ang = [15,30,45,60,90,120]
        fig, axs = plt.subplots(3,1)
        
        
        cmap = matplotlib.cm.get_cmap('magma')
        
        rgba = cmap(0.5)
        
        fig.canvas.draw()
        
        
        
        
        
        
        df = pd.DataFrame(list(zip(labels,semi_major_list, semi_minor_list, area_list, ratio_list, distance_right, azimuth_list, altitude_list )), columns =['Visual_angle_and_orientation',
                                                                                             'Semi major',
                                                                                             'Semi minor',
                                                                                             'Image_area',
                                                                                             'Aspect ratio',
                                                                                             'Distance to retina center',
                                                                                             'Azimuth',
                                                                                             'Altitude'])
         
        df.to_csv (filename, index = False, header=True)
        
        return 
