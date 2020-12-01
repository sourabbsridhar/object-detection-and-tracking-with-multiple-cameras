# Optical Flow

# Input vector:
# Folder with textfiles in the format:
# [time, id , x, y (top left corner), width, height ,confidence score, class, visibility, ???]

# Output parameters: 
# state = [frame, id, x y dx dy][pixels/frame]

import cv2
import numpy as np

object_all = [] # empty list to add lines in
object_id = [] # extract one id
box_velocity = [] # List with all velocities of the bounding boxes
state = [] # State [x y dx dy] [pxls/frame]
numbers = [] # List with all ID numbers in increasing order

with open("optical-flow/optical_train/MOT16-02.txt", "r") as f: # open while in this indent
    
    f.readline() # read lines
    for index,line in enumerate(f):
        object_all.append(line.split(','))
                
# Print unsorted list            
#print('unsorted list')            
#for i in object_all:
#    print(i)

# Sort list by ID
object_all.sort(key=lambda x: int(x[1]))

# Print sorted list
#print('sorted list')
#for j in object_all:
#    print(j)


# Add unique ID numbers from input file to number list
for index,line in enumerate(object_all):
    if line[1] not in numbers:
        numbers.append(line[1]) 

# Extract id rows from input and calulate state
for identity in numbers: # Loop trough every ID found
    object_id = [] # reset list for every new ID
    for index,line in enumerate(object_all): # Extract all lines with current ID
        if line[1] == identity: # line[1] = ID column
            object_id.append(line) # Add ID lines to list

    # Reset parameters for every new ID
    first = True 
    box = []
    box_prev = []

    # Extract bounding box coordinates
    for b in range(len(object_id)): # Loop all lines in ID list
        if len(object_id) < 2: # If less than 2 bounding boxes, calculations can not be done, thus break the loop
            print('Could not track velocity, not enough bounding boxes!')
            break
        else:
            box_all = [float(i) for i in object_id[b]] # Convert values in list from str to float
            if first: # If first bounding box, set it to box_prev
                first = False 
                box_prev = box_all[2:6] # Extract bounding box coordinates
            else:
                box = box_all[2:6]  # Extract bounding box coordinates
            
                # Center point for previous box
                box_prev_center_x = box_prev[0] + box_prev[2]/2
                box_prev_center_y = box_prev[1] - box_prev[3]/2
                box_prev_center = np.array([box_prev_center_x,box_prev_center_y])

                # Center point for current box
                box_center_x = box[0] + box[2]/2
                box_center_y = box[1] - box[3]/2
                box_center = np.array([box_center_x,box_center_y])

                # Box pixels change, vector of delta x and delta y 
                box_delta_xy = box_prev_center - box_center

                # Distance between center points of box and box_prev
                #box_distance = np.sqrt(np.power(box_delta_xy[0],2) + np.power(box_delta_xy[1],2)) # Norm

                state.append([int(box_all[0]), identity, box_center_x, box_center_y, box_delta_xy[0], box_delta_xy[1]])

                box_prev = box # Update the the current box to the previous box

# Print states 
#for frame in state:
    #print('object id \n',object_id)
    #print('box position and velocity', frame)


# Write states to the txt file states_multiple.txt
with open("optical-flow/tracked_objects/states_multiple.txt", "w") as fs:
    for line in state:
            fs.write(str(line) + "\n")

    





