# Main implementation of the overall software

import argparse
from pathlib import Path

import code.object_detection as objDet
import code.pose_estimation as poseEst
import code.project_2d_position as proj2dPos
import code.fuse_2d_position as fusePos

def evaluateArguments(inputPath, detectionWeightsPath, trackingWeightsPath, cameraParametersPath, outputPath, verbose):

    evaluationSuccessful = True
    if verbose:
        print("Evaluating Input Arguments...")
    if not (Path.exists(inputPath)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Input Path Does Not Exist (Path: {}). Aborting Execution!".format(inputPath))
    elif not (Path.exists(detectionWeightsPath)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Object-Detection Weights Does Not Exist (Path: {}). Aborting Execution!".format(detectionWeightsPath))
    elif not (Path.exists(trackingWeightsPath)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Object-Tracking Weights Does Not Exist (Path: {}). Aborting Execution!".format(trackingWeightsPath))
    elif not (Path.exists(cameraParametersPath)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Camera Parameters Does Not Exist (Path: {}). Aborting Execution!".format(cameraParametersPath))
    elif not (Path.exists(outputPath)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Output Path Does Not Exist (Path: {}). Aborting Execution!".format(outputPath))
    else:
        if verbose:
            print("[Log] Evaluation of Input Arguments Successful!")
    
    return evaluationSuccessful

def evaluateSoftware(verbose):
    
    return True

def excecuteSoftware(inputPath, detectionWeightsPath, trackingWeightsPath, cameraParametersPath, outputPath, verbose):

    for frameID in range(1):
        objDet.displayInputFrame(inputPath, frameID)
        detections2dPosition = objDet.objectDetection(inputPath, frameID)
        estimatedPose = poseEst.estimatePose(cameraParametersPath)
        projection3dPosition = proj2dPos.project2dPosition(detections2dPosition, estimatedPose)
        proj2dPos.display3dPosition(projection3dPosition)
        fused3dPosition = fusePos.fuse3dPosition(projection3dPosition)
        proj2dPos.display3dPosition(fused3dPosition)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-object Detection and Tracking using Multiple Cameras")
    parser.add_argument("-i", "--input", help="path to input folder", default=Path.cwd())
    parser.add_argument("-wd", "--weightsdetection", help="path to object-detection weights", default=Path.cwd())
    parser.add_argument("-wt", "--weightstracking", help="path to object-tracking weights", default=Path.cwd())
    parser.add_argument("-cp", "--cameraparameters", help="path to camera parameters", default=Path.cwd())
    parser.add_argument("-o", "--output", help="path to output folder", default=Path.cwd())
    parser.add_argument("-v", "--verbose", help="display log", action="store_true")
    args = parser.parse_args()

    inputPath = args.input
    detectionWeightsPath = args.weightsdetection
    trackingWeightsPath = args.weightstracking
    cameraParametersPath = args.cameraparameters
    outputPath = args.output
    verbose = args.verbose

    if verbose:
        
        print("**********************************************************")
        print("Multi-object Detection and Tracking using Multiple Cameras")
        print("**********************************************************")
        print("Path to Input Folder: {}".format(inputPath))
        print("Path to object-detection weights: {}".format(detectionWeightsPath))
        print("Path to object-tracking weights: {}".format(trackingWeightsPath))
        print("Path to camera parameters: {}".format(cameraParametersPath))
        print("Path to output folder: {}".format(outputPath))

    areArgumentsValid = evaluateArguments(inputPath, detectionWeightsPath, trackingWeightsPath, cameraParametersPath, outputPath, verbose)
    isSoftwareValid = evaluateSoftware(verbose)

    if ((areArgumentsValid == True) and (isSoftwareValid == True)):
        excecuteSoftware(inputPath, detectionWeightsPath, trackingWeightsPath, cameraParametersPath, outputPath, verbose)
    else:
        print("[ERROR] Aborting Software Execution!")