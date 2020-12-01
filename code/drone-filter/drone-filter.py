import cv2
import numpy as np
import os 
import glob
import csv
from random import random

output = []
lines = []

#PATH = "/Users/divyakara/Documents/Chalmers/Master/Year2/SSY226- Design_Project/GitLab/object-detection-and-tracking-with-multiple-cameras/code/drone-filter/annotations/"
# Path for input data
PATH = "./code/drone-filter/annotations/"

# Folder for output data
PATH_out = "code/drone-filter/generated-detections"

# All paths in input folder
PATH_in_all = glob.glob(PATH+'*/*/*.txt')

# Empty list to fill with all output paths
PATH_out_all = []

# Loop over all input paths to read data files
for path in PATH_in_all:
    # Open all input files
    with open(path, "r") as f: # open while in this indent

        # Read lines in file
        f.readline() 

        # Loop over all lines in input text file
        for index,line in enumerate(f):
            # Split lines by space and save to lines
            lines = line.split(' ') 
            
            # Add noise - Skip some lines
            PERCENTAGE_NOISE = 0.05
            if random() < PERCENTAGE_NOISE:
                continue

            # Check if line/object is not occluded
            if int(lines[7]) != 1:
                # if line not occluded add to output list, occluded objects are removed
                output.append(lines)

            # set ID to -1 
            lines[1] = -1
            
            # Dictonary class
            name_dict = {}
            name_dict["Biker"] = 0
            name_dict["Pedestrian"] = 1
            name_dict["Skater"] = 2 
            name_dict["Cart"] = 3
            name_dict["Car"] = 4
            name_dict["Bus"]  = 5

            # Put all unique objects in a list
            obj = str(lines[-1])
            obj = obj.split('"')[1] # Remove "" and \n 
            #print("Object", obj)

            # Move to pos -2
            lines[-2] = name_dict[obj]

            # Visibility = 1 new pos
            lines[-1] = 1
        
        # Add subfolders to output folder, same structure as input folder
        PATH_out = str.replace(path, 'annotations','generated_detections')

        # Create directory if it does not exist
        if not os.path.isdir(PATH_out):
            os.makedirs(os.path.dirname(PATH_out), exist_ok=True)
            
        # Write to output file    
        with open(PATH_out, "w") as fs:
            for line in output:
                # Write comma seperated file
                fs.write(','.join(str(i) for i in line))
                # New line after each object
                fs.write('\n')

    print('#### ONE PATH DONE! ###', PATH_out)
print('##### ALL DONE! ######')