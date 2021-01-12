# Implementation of the Pose Estimation Function (To be provided by the handover subgroup)

import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET

def estimatePose(cameraParametersPath):

    intrinsicParametersPath = str(cameraParametersPath) + "\\**\\intrinsic_zero\\**\\*.xml"
    intrinsicParametersPath = glob.glob(intrinsicParametersPath, recursive=True)
    intrinsicParametersPath = sorted(intrinsicParametersPath)

    extrinsicParametersPath = str(cameraParametersPath) + "\\**\\extrinsic\\**\\*.xml"
    extrinsicParametersPath = glob.glob(extrinsicParametersPath, recursive=True)
    extrinsicParametersPath = sorted(extrinsicParametersPath)

    cameraParameters = list()

    #TODO: Add Assert Check, Fix XML Format Issue (I have fixed the format of extrinsic parameters manually!!), Check Logic (Code is Wrong)!!!

    for cameraIndex in range(len(extrinsicParametersPath)):

        currentIntrinsicCameraParameters = ET.parse(intrinsicParametersPath[cameraIndex])
        currentIntrinsicCameraParameters = currentIntrinsicCameraParameters.find("camera_matrix")
        currentIntrinsicCameraParameters = currentIntrinsicCameraParameters.find("data")
        intrinsicParameters = np.array(currentIntrinsicCameraParameters.text.split(" "))
        intrinsicParameters = np.reshape(intrinsicParameters, (3,3))
        intrinsicParameters = intrinsicParameters.astype('float64')

        currentExtrinsicCameraParameters = ET.parse(extrinsicParametersPath[cameraIndex])
        currentExtrinsicCameraParametersRotation = currentExtrinsicCameraParameters.find("rvec")
        extrinsicRotationParameters = np.array(currentExtrinsicCameraParametersRotation.text.split(" "))
        extrinsicRotationParameters = np.reshape(extrinsicRotationParameters, (3,1))
        extrinsicRotationParameters = extrinsicRotationParameters.astype('float64')
        extrinsicRotationParameters = np.identity(3)
        #extrinsicRotationParameters = np.matmul(np.identity(3), extrinsicRotationParameters)
        #extrinsicRotationParameters = cv2.Rodrigues(extrinsicRotationParameters) (To be Checked)

        currentExtrinsicCameraParameters = ET.parse(extrinsicParametersPath[cameraIndex])
        currentExtrinsicCameraParametersTranslation = currentExtrinsicCameraParameters.find("tvec")
        extrinsicTranslationParameters = np.array(currentExtrinsicCameraParametersTranslation.text.split(" "))
        extrinsicTranslationParameters = np.reshape(extrinsicTranslationParameters, (3,1))
        extrinsicTranslationParameters = extrinsicTranslationParameters.astype('float64')

        extrinsicParameters = np.append(extrinsicRotationParameters, extrinsicTranslationParameters, axis=1)
        currentCameraParameters = np.matmul(intrinsicParameters, extrinsicParameters)
        cameraParameters.append(currentCameraParameters)        


    return cameraParameters