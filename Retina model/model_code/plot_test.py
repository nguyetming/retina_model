
from Eye_model_function_curve import *
from additional_input import *
import scipy.io as sp
from numpy import genfromtxt
import pandas as pd


###INDICATE LEFT Or 
#PUT IN LOCATION OF THE DATA FILES:
location1 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_ver_1_LeftData.xlsx'
location2 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_ver_2_LeftData.xlsx'
location3 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_ver_3_LeftData.xlsx'
location4 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_hor_1_LeftData.xlsx'
location5 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_hor_2_LeftData.xlsx'
location6 = r'/Users/nguyetnguyen/Downloads/Retina model/Retina model/model_code/image_constant_hor_3_LeftData.xlsx'

location = [ location3, location2, location1, location6, location5, location4]


#COLOR OF IMAGES
colors = ['red', 'green', 'blue','orange','black', 'yellow','pink']
fig = plt.figure()
ax0 = fig.add_subplot(1,2,1)

ax1 = fig.add_subplot(1,2,2, projection='3d')
for i in range(len(location)):
    
    # LOAD DATA FRAME AND THE COLUMN YOU WANT TO USE FROM THE DATAFRAME
    df = pd.read_excel(location[i], usecols=[0,1,2,3,4,5],  engine='openpyxl')
    
    cmap = matplotlib.cm.get_cmap('magma')
    x_image_list = df.iloc[:,3].to_list()
    y_image_list = df.iloc[:,4].to_list()
    z_image_list = df.iloc[:,5].to_list()
    x_list = df.x_point.to_list()
    y_list = df.y_point.to_list()
    z_list = df.z_point.to_list()
    
    
    #plot image on the retina
    [xr,yr,zr] = right_eye_wire
    [xl, yl, zl] = left_eye_wire
    ax1.plot_wireframe(xr, yr, zr, color="lightblue")
    ax1.plot_wireframe(xl, yl, zl, color="lightblue")
    ax1.plot(x_image_list, y_image_list, z_image_list, marker='o', color = colors[i])
    
    ax1.scatter(x_center_retina_left * arena_radius, y_center_retina_left * arena_radius, h*arena_radius, marker='o', s = 10, color = 'red')
    ax1.scatter(x_center_retina_right * arena_radius, y_center_retina_right * arena_radius, h* arena_radius, marker='o', s = 10, color = 'red')
    
    print("x,y,z retina center:", x_center_retina_left * arena_radius, y_center_retina_left * arena_radius, h*arena_radius)
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    print("x list len:", np.size(x_list), np.size(y_list))
    
    #plot point shown to the fish
    ax0.grid(False)
    ax0.scatter(x_list, y_list, s = 5, marker='o', color = colors[i])
    ax0.scatter(x_leye*arena_radius, y_leye*arena_radius, marker='o', s = 5, color = 'r')
    ax0.scatter(x_reye*arena_radius, y_reye*arena_radius, marker='o', s = 5, color = 'r')
    ax0.set_aspect('equal')
    
    
plt.show()
#plt.savefig('demo.png', transparent=True)      
# #         camera.snap()
        
# #     animation = camera.animate()
          

# # #### delete the animation. 
