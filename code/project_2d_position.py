# Implementation of Projection Function to project 2D Image points onto a 3D plane (To be provided by the handover subgroup)

import cv2
import numpy as np
from . import object_detection
from . import Handover_Fusion
from . import Simulation

def projections(imagePoints, cameras, groundHeight):
    projections = list()

    for imagePoint in imagePoints:
        camera = next((camera for camera in cameras if camera.id == imagePoint.camera_id), None)

        if camera is None:
            raise Exception('There is no camera found for image point camera id')

        # Homogenous coordinates
        position2dHomogenous = np.vstack(imagePoint.position, 1)
        velocity2dHomogenous = np.vstack(imagePoint.velocity, 1)

        P_tilde = np.matmul(np.linalg.inv(camera.K), position2dHomogenous)
        V_tilde = np.matmul(np.linalg.inv(camera.K), position2dHomogenous)

        l = (groundHeight - np.matmul(camera.R[:, 3].T, camera.t)) / np.matmul(camera.R[:, 3].T, P_tilde)

        P_bar = P_tilde * l
        V_bar = V_tilde * l

        p_bar = pflat(P_bar)[0:1]
        v_bar = pflat(V_bar)[0:1]

        projection = Projection(imagePoint.detection_id, imagePoint.camera_id, imagePont.detection_class, p_bar, v_bar)
        projections.append(projection)


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
            projected3dHomogenous = projected3dHomogenous/projected3dHomogenous[3, 0]
            projected3dHomogenous = projected3dHomogenous[0:3]

            #print("x = ", str(x))
            #print("y = ", str(y))
            #print("position2dHomogenous = ", str(position2dHomogenous))
            #print("currentCameraMatrix = ", str(currentCameraMatrix))
            #print("projected3dHomogenous = ", str(projected3dHomogenous))


    return 1
            


def display3dPosition(projection3dPosition):
    pass
