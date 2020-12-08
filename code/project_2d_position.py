# Implementation of Projection Function to project 2D Image points onto a 3D plane (To be provided by the handover subgroup)

import cv2
import numpy as np

from . import object_detection

def project2dPosition(detections2dPosition, estimatedPose):

    for current2dPositionDetected in detections2dPosition:

        print("current2dPositionDetected = " + str(current2dPositionDetected))
        print(type(current2dPositionDetected))

        for cameraIndex in range(len(estimatedPose)):
            #print(current2dPositionDetected.cameraViews[cameraIndex])
            #TODO: Current the issues with input files (detection) are causing issues, needs to be fixed!!
            # For every detection and every camera view, project the 2D image point to 3D point

            x = float(current2dPositionDetected.cameraViews[cameraIndex]["x"])
            y = float(current2dPositionDetected.cameraViews[cameraIndex]["y"])
            position2dHomogenous = np.array([[x], [y], [1]])
            currentCameraMatrix = estimatedPose[cameraIndex]

            projected3dHomogenous = np.matmul(np.linalg.pinv(currentCameraMatrix), position2dHomogenous)
            projected3dHomogenous = projected3dHomogenous/projected3dHomogenous[3,0]
            projected3dHomogenous = projected3dHomogenous[0:3]

            #print("x = ", str(x))
            #print("y = ", str(y))
            #print("position2dHomogenous = ", str(position2dHomogenous))
            #print("currentCameraMatrix = ", str(currentCameraMatrix))
            #print("projected3dHomogenous = ", str(projected3dHomogenous))


    return 1
            


def display3dPosition(projection3dPosition):
    pass
