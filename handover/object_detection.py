# Implementation of Object Detection Algorithm (To be provided by the object detection subgroup)

import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Detections:
    def __init__(self, objectID, objectClass, objectPosition):
        self.objectID = objectID
        self.objectClass = objectClass
        self.objectPosition = objectPosition
        self.cameraViews = list()

    def __repr__(self):
        outputString = "\nDetections(objectID = {}, objectClass = {}, objectPosition = {}, cameraViews = {})\n".format(self.objectID,\
             self.objectClass, self.objectPosition, self.cameraViews)
        return outputString

    def setCameraViews(self, cameraViews):
        self.cameraViews = cameraViews

class CameraFrame():
    def __init__(self, frame):
        self.frame = frame
        self.overallDetections = list()

    def __str__(self):
        outputString = ""
        for detectionIndex in range(len(self.overallDetections)):
            outputString = outputString + "\n{}\n".format(self.overallDetections[detectionIndex])
        return outputString

    def getDetections(self):
        with open(self.frame) as f:
            overallData = json.load(f)
            for individualObject in overallData:
                currentDetection = Detections(individualObject["personID"], -1, individualObject["positionID"])
                for individualCameraView in individualObject["views"]:
                    individualCameraView["x"] = (individualCameraView["xmin"] + individualCameraView["xmax"])/2
                    individualCameraView["y"] = (individualCameraView["ymin"] + individualCameraView["ymax"])/2
                currentDetection.setCameraViews(individualObject["views"])
                self.overallDetections.append(currentDetection)
        return self.overallDetections            

def displayInputFrame(inputPath, frameID):
    cameraFoldersName = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    overallCameraImagesPath = list()

    for individualCameraFolderName in cameraFoldersName:
        individualCameraImagesPath = str(inputPath) + "\\**\\" + individualCameraFolderName + "\\**\\*.png"
        individualCameraImagesPath = glob.glob(individualCameraImagesPath, recursive=True)
        individualCameraImagesPath = sorted(individualCameraImagesPath)
        overallCameraImagesPath.append(individualCameraImagesPath)

    inputImagePath = list()
    for camerapersonIndex in range(len(cameraFoldersName)):
        inputImagePath.append(overallCameraImagesPath[camerapersonIndex][frameID])

    inputImage = list()
    for camerapersonIndex in range(len(cameraFoldersName)):
        inputImage.append(mpimg.imread(inputImagePath[camerapersonIndex]))
    
    #TODO: Improve This Code
    _, axs = plt.subplots(3, 3, figsize=(8,8))
    axs[0, 0].imshow(inputImage[0])
    axs[0, 1].imshow(inputImage[1])
    axs[0, 2].imshow(inputImage[2])
    axs[1, 1].imshow(inputImage[3])
    axs[2, 0].imshow(inputImage[4])
    axs[2, 1].imshow(inputImage[5])
    axs[2, 2].imshow(inputImage[6])
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def objectDetection(inputPath, frameID):

    annotationPositionsPath = str(inputPath) + "\\**\\annotations_positions\\**\\*.json"
    annotationPositionsPath = glob.glob(annotationPositionsPath, recursive=True)
    annotationPositionsPath = sorted(annotationPositionsPath)

    individualFrame = CameraFrame(annotationPositionsPath[frameID])
    individualFrameDetections = individualFrame.getDetections()

    #print(individualFrameDetections)

    return individualFrameDetections
