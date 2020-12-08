import cv2
import numpy as np
import os 
import glob
import csv
from random import random
from random import randint


'''
    1   Track ID. All rows with the same ID belong to the same path.
    2   xmin. The top left x-coordinate of the bounding box.
    3   ymin. The top left y-coordinate of the bounding box.
    4   xmax. The bottom right x-coordinate of the bounding box.
    5   ymax. The bottom right y-coordinate of the bounding box.
    6   frame. The frame that this annotation represents.
    7   lost. If 1, the annotation is outside of the view screen.
    8   occluded. If 1, the annotation is occluded.
    9   generated. If 1, the annotation was automatically interpolated.
    10  label. The label for this annotation, enclosed in quotation marks.
'''
# Define Image size to generate false detections at
IM_WIDTH = 1424
IM_HEIGHT = 1088
#PATH = "/Users/divyakara/Documents/Chalmers/Master/Year2/SSY226- Design_Project/GitLab/object-detection-and-tracking-with-multiple-cameras/code/drone-filter/annotations/"
# Path for input data
PATH = "/home/jonatan/SSY226/object-detection-and-tracking-with-multiple-cameras/code/drone-filter/annotations/"

# All paths in input folder
PATH_in_all = glob.glob(PATH+'*/*/*.txt')
#print(PATH_in_all)

# Loop over all input paths to read data files
for path in PATH_in_all:
    # Folder for output data
    PATH_out = "/home/jonatan/SSY226/object-detection-and-tracking-with-multiple-cameras/dataset/stanford_drone/generated-detections/"
    # Open all input files
    with open(path, "r") as f: # open while in this indent

        # Read lines in file
        f.readline() 
        output = []

        # Loop over all lines in input text file
        for index,line in enumerate(f):
            # Split lines by space and save to lines
            lines = line.split(' ') 
            
            # Add noise - Skip some lines
            PERCENTAGE_NOISE = 0.1
            if random() < PERCENTAGE_NOISE:
                continue
                
            
            # Skip if no bounding box
            #if int(lines[2]) == 0:
            #    continue

            # Check if line/object is not occluded
            if int(lines[7]) == 1:
                # if line occluded skip the row, ocluded detections are ignored
                continue

            
            # Define all annotation parameters
            ID, xmin, ymin, xmax, ymax, frame, lost, ocluded, generated, label = lines
            
            # Dictonary class
            name_dict = {}
            name_dict["Biker"] = 0
            name_dict["Pedestrian"] = 1
            name_dict["Skater"] = 2 
            name_dict["Cart"] = 3
            name_dict["Car"] = 4
            name_dict["Bus"]  = 5

            # Put all unique objects in a list
            obj = str(label)
            obj = obj.split('"')[1] # Remove "" and \n 

            # Change the output to correct format and give labels a number
            output.append([frame, -1, xmin, ymin, xmax, ymax, np.max(0.5, random()), name_dict[obj], 1, -1]) # Random since confidence is different

            # Add false positives - Add some false detections
            PERCENTAGE_FP = 0.1
            if random() < PERCENTAGE_FP:
                x1 = IM_WIDTH*random()
                y1 = IM_HEIGHT*random()
                x2 = x1 + 70*random()
                y2 = y1 + 100*random()
                output.append([frame, -1, x1, y1, x2, y2, random(), randint(0,5), 1, -1])
        
        # Sort the list with respect to the frames

        output.sort(key=lambda x : int(x[0]))

        # Add subfolders to output folder, same structure as input folder
        PATH_out += path.split('/')[-3]
        PATH_out += '/'
        PATH_out += path.split('/')[-2]
        PATH_out += '/det/'
        #print(PATH_out)
        #break

        # Create directory if it does not exist
        if not os.path.isdir(PATH_out):
            os.makedirs(os.path.dirname(PATH_out), exist_ok=True)
            print('Created path:', PATH_out)
        
        PATH_out += 'det.txt'

        # Write to output file    
        with open(PATH_out, "w") as fs:
            for line in output:
                # Write comma seperated file
                fs.write(','.join(str(i) for i in line))
                fs.write('\n')

    print('#### ONE PATH DONE! ###', PATH_out)
print('##### ALL DONE! ######')