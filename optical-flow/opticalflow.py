###### Start up ########
# conda install 'package'
# conda activate optical-flow

#########################
# Input vector:
# Folder with textfiles in the format:
# [time, id , x, y (top left corner), width, height ,confidence score, class, visibility, ???]

# Output parameters: 
# state = [x y dx dy][pixels/frame]

#########################
# Tips and trix:
# str.strip('x')  = ta bort tecken (x)
# 

import cv2
import numpy as np

print('##############################################################################################################')

object_all = [] # empty list to add lines in
object_id = [] # extract one id
box_velocity = [] # List with all velocities of the bounding boxes
state = [] # State [x y dx dy] [pxls/frame]
id = 10

with open("optical-flow/optical_train/MOT16-02.txt", "r") as f: # open while in this indent
    
    f.readline() # read lines
    for index,line in enumerate(f):
        object_all.append(line.split(','))
                
            
#print('unsorted list')            
#for i in object_all:
#    print(i)

# Sort list by ID
object_all.sort(key=lambda x: int(x[1]))

print('sorted list')
for j in object_all:
    print(j)


# Extract lines from input files with id = 10
for index,line in enumerate(object_all):
    if (line[1] == str(id) ):
        object_id.append(line) 
        




# Extract bounding box coordinates
first = True
for b in range(len(object_id)):
    box_all = [float(i) for i in object_id[b]] #if i mod(2)
    if first: 
        first = False 
        box_prev = box_all[2:6] 
    else:
        box = box_all[2:6] 
    
        # Center point for previous box
        box_prev_center_x = box_prev[0] + box_prev[2]/2
        box_prev_center_y = box_prev[1] - box_prev[3]/2
        box_prev_center = np.array([box_prev_center_x,box_prev_center_y])

        # Center point for current box
        box_center_x = box[0] + box[2]/2
        box_center_y = box[1] - box[3]/2
        box_center = np.array([box_center_x,box_center_y])

        # Box pixels change 
        box_change_xy = box_prev_center - box_center

        # Distance between center points of boxes
        box_distance = np.sqrt(np.power(box_change_xy[0],2) + np.power(box_change_xy[1],2))
        print('box distance', box_distance)

        print('\nbounding box prev:', box_prev)
        print('bounding box:', box)
        box_prev = box # Update the the current box to the previous box

        box_velocity.append(box_distance)
        print('box_change:', box_change_xy)

        state.append([box_center_x, box_center_y, box_change_xy[0], box_change_xy[1]])

#print('box_velocity:', (np.array(box_velocity)))
for frame in state:
    print('box position and velocity', frame)


# States added to the txt file states
#outF = open("optical-flow/tracked_objects/states_1D.txt", "w")
#for line in state:
#    outF.write(str(line))
#    outF.write("\n")
#outF.close()
    
with open("optical-flow/tracked_objects/states_1D.txt", "w") as fs: # behöver inte close
    for line in state:
        fs.write(str(line) + "\n")
        #fs.write("\n")
    










