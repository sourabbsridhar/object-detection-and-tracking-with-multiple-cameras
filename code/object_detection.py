# Implementation of Object Detection Algorithm (To be provided by the object detection subgroup)

import cv2
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class detectedObject:
    def __init__(self, frame):
        self.frame = frame
        self.personID = list()
        self.positionID = list()
        self.views = list()

    def __str__(self):
        outputString = ""
        for index in range(len(self.personID)):
            outputString = outputString + "\n\nPerson ID: {}\nPosition ID: {}\nViews:\n{}".format(self.personID[index], self.positionID[index], self.views[index])
        return outputString

    def getDetections(self):
        with open(self.frame) as f:
            overallData = json.load(f)
            for person in overallData:
                self.personID.append(person["personID"])
                self.positionID.append(person["positionID"])
                for camera_view in person["views"]:
                    camera_view["x"] = (camera_view["xmin"] + camera_view["xmax"])/2
                    camera_view["y"] = (camera_view["ymin"] + camera_view["ymax"])/2
                self.views.append(person["views"])
        return self
                

def display_input(input_path, frameID):
    camera_folders = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    camera_images = list()

    for folder_name in camera_folders:
        current_camera_images = str(input_path) + "\\**\\" + folder_name + "\\**\\*.png"
        current_camera_images = glob.glob(current_camera_images, recursive=True)
        current_camera_images = sorted(current_camera_images)
        camera_images.append(current_camera_images)

    img_path = list()
    for camera_index in range(len(camera_images)):
        img_path.append(camera_images[camera_index][frameID])

    fig, axs = plt.subplots(3, 3)
    img1 = mpimg.imread(img_path[0])
    img2 = mpimg.imread(img_path[1])
    img3 = mpimg.imread(img_path[2])
    img4 = mpimg.imread(img_path[3])
    img5 = mpimg.imread(img_path[4])
    img6 = mpimg.imread(img_path[5])
    img7 = mpimg.imread(img_path[6])
    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[0, 2].imshow(img3)
    axs[1, 1].imshow(img4)
    axs[2, 0].imshow(img5)
    axs[2, 1].imshow(img6)
    axs[2, 2].imshow(img7)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def object_detection(input_path, frameID):

    annotation_positions = str(input_path) + "\\**\\annotations_positions\\**\\*.json"
    annotation_positions = glob.glob(annotation_positions, recursive=True)
    annotation_positions = sorted(annotation_positions)

    obj = detectedObject(annotation_positions[frameID])
    detections = obj.getDetections()

    return detections
    

