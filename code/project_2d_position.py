# Implementation of Projection Function to project 2D Image points onto a 3D plane (To be provided by the handover subgroup)

import cv2
import object_detection

def project2dPosition(detections2dPosition, estimatedPose):

    for current2dPositionDetected in detections2dPosition:
        for cameraIndex in range(len(estimatedPose)):
            #TODO: Current the issues with input files (detection) are causing issues, needs to be fixed!!
            # For every detection and every camera view, project the 2D image point to 3D point
            print(cameraIndex)


def display3dPosition(projection3dPosition):
    pass
