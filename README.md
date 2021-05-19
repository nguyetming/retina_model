# Retina model
> A pin-hole model of fish's retina which transforms bottom-projected stimuli into retinal space

## Examples of retina images 
![Example image](./retina_image.png)

## Setup
* Install the latest version of Python 3. We are using the Anaconda distribution
* Some additional packages needed: 
    * skimage - an open source Python package designed for image preprocessing; 
      ```
      # Update pip
      python -m pip install -U pip
      # Install scikit-image
      python -m pip install -U scikit-image
      ```   
    * openpyxl  - a Python library to read/write Excel 2010 xlsx/xlsm/xltx/xltm files; 
      ```
      pip install openpyxl      
      ```
   
    * pandas - a software library written for the Python programming language for data manipulation and analysis; 
      ```
      pip install pandas
      ```

## Usage
* Put in parameters for the visual stimuli as follow:
   ```
   ratio = 1### how big is one axis compared to the other.
   minor_p = 0.225  #minor axis of the dot   (cm)
   major_p = minor_p * ratio #major axis of the dot  (cm)
   dist_to_center =  [0.9]   # distance from the fish's head to the center of the stimulus (cm) 
   angle_to_fish = [20] #angle between Oy and the line connecting the closet point of the ellipse to the fish
   angle_ver = [20]  # angle from Oy to the major axis of the ellipse  
   location = r'WHERE YOU WANT TO SAVE THE OUTPUT DATA FILES'
   labels = "vertical_constant" ##labels of stimulus. Eg: vertical ellipse or horizontal ellipse
   ```
   
