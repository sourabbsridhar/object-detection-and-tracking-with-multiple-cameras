# Implementation of Projection Function to project 2D Image points onto a 3D plane (To be provided by the handover subgroup)

#import cv2
import numpy as np
#from . import object_detection
#from . import Handover_Fusion
#from . import Simulation
from Handover_Library import Projection, pflat

def ground_projections(imagePoints, cameras, groundHeight, deltaTime):

    projections = list()
    for imagePoint in imagePoints:
        camera = next((camera for camera in cameras if camera.id == imagePoint.camera_id), None)

        if camera is None:
            raise Exception('There is no camera found for image point camera id')

        # Calculate current ground position
        p_tilde = np.vstack((imagePoint.position, 1))
        P_tilde = np.matmul(np.linalg.inv(camera.K), p_tilde)
        lamb = (groundHeight + np.matmul(camera.R[:, 2].T, camera.t)) / np.matmul(camera.R[:, 2].T, P_tilde)
        P_bar = P_tilde * lamb
        P_bar_global = np.matmul(camera.R.T, P_bar - camera.t)
        p_bar = P_bar_global[0:2]

        # If there is a previous image point position available, calculate its projection position and use it to
        # calculate the estimated projected velocity, otherwise the velocity will be zero
        v_bar = np.zeros((2, 1))
        if imagePoint.prev_position is not None:
            p_tilde = np.vstack((imagePoint.prev_position, 1))
            P_tilde = np.matmul(np.linalg.inv(camera.K), p_tilde)
            lamb = (groundHeight + np.matmul(camera.R[:, 2].T, camera.t)) / np.matmul(camera.R[:, 2].T, P_tilde)
            P_bar = P_tilde * lamb
            P_bar_global = np.matmul(camera.R.T, P_bar - camera.t)
            prev_p_bar = P_bar_global[0:2]

            v_bar = (p_bar - prev_p_bar) / deltaTime

        projection = Projection(imagePoint.detection_id, imagePoint.camera_id, imagePoint.detection_class, p_bar, v_bar)
        projections.append(projection)

    return projections


    # Ground projection which calculated ground velocity from image point velocity
    '''
    projections = list()
    for imagePoint in imagePoints:
        camera = next((camera for camera in cameras if camera.id == imagePoint.camera_id), None)

        if camera is None:
            raise Exception('There is no camera found for image point camera id')

        # Homogenous coordinates
        position2dHomogenous = np.vstack((imagePoint.position, 1))
        velocity2dHomogenous = np.vstack((imagePoint.velocity, 1))

        P_tilde = np.matmul(np.linalg.inv(camera.K), position2dHomogenous)
        testK = np.hstack((camera.K[:, 0:2], np.array([[0], [0], [1]])))
        V_tilde = np.matmul(np.linalg.inv(testK), velocity2dHomogenous)

        l = (np.matmul(camera.R[:, 2].T, camera.t) - groundHeight) / np.matmul(camera.R[:, 2].T, P_tilde)

        P_bar = P_tilde * l
        V_bar = V_tilde * l

        P_bar_global = np.matmul(camera.R.T, P_bar - camera.t)
        V_bar_global = np.matmul(camera.R.T, V_bar - camera.t)

        p_bar = P_bar_global[0:2]
        v_bar = V_bar_global[0:2]

        projection = Projection(imagePoint.detection_id, imagePoint.camera_id, imagePoint.detection_class, p_bar, v_bar)
        projections.append(projection)

    return projections
    '''

'''
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
'''