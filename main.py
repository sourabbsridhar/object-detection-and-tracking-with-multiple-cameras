# Main implementation of the overall software

import argparse
from pathlib import Path

def evaluateArguments(input_path, detection_weights_path, tracking_weights_path, camera_parameters_path, output_path, verbose):

    evaluationSuccessful = True

    if verbose:
        print("Evaluating Input Arguments...")

    if not (Path.exists(input_path)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Input Path Does Not Exist (Path: {}). Aborting Execution!".format(input_path))
    elif not (Path.exists(detection_weights_path)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Object-Detection Weights Does Not Exist (Path: {}). Aborting Execution!".format(detection_weights_path))
    elif not (Path.exists(tracking_weights_path)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Object-Tracking Weights Does Not Exist (Path: {}). Aborting Execution!".format(tracking_weights_path))
    elif not (Path.exists(camera_parameters_path)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Path to Camera Parameters Does Not Exist (Path: {}). Aborting Execution!".format(camera_parameters_path))
    elif not (Path.exists(output_path)):
        evaluationSuccessful = False
        if verbose:
            print("[Error] Output Path Does Not Exist (Path: {}). Aborting Execution!".format(output_path))
    else:
        if verbose:
            print("[Log] Evaluation of Input Arguments Successful!")
    
    return evaluationSuccessful

def evaluateSoftware(verbose):
    
    return True

def excecuteSoftware(input_path, detection_weights_path, tracking_weights_path, camera_parameters_path, output_path, verbose):

    """
    Object detection
    Pose estimation
    Object tracking
    Velocity estimation
    Project 2D position
    Project 2D velocity
    Fuse Multi-Camera Detection
    """
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-object Detection and Tracking using Multiple Cameras")
    parser.add_argument("-i", "--input", help="path to input folder", default=Path.cwd())
    parser.add_argument("-wd", "--weightsdetection", help="path to object-detection weights", default=Path.cwd())
    parser.add_argument("-wt", "--weightstracking", help="path to object-tracking weights", default=Path.cwd())
    parser.add_argument("-cp", "--cameraparameters", help="path to camera parameters", default=Path.cwd())
    parser.add_argument("-o", "--output", help="path to output folder", default=Path.cwd())
    parser.add_argument("-v", "--verbose", help="display log", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    detection_weights_path = args.weightsdetection
    tracking_weights_path = args.weightstracking
    camera_parameters_path = args.cameraparameters
    output_path = args.output
    verbose = args.verbose

    if verbose:
        
        print("**********************************************************")
        print("Multi-object Detection and Tracking using Multiple Cameras")
        print("**********************************************************")
        print("Path to Input Folder: {}".format(input_path))
        print("Path to object-detection weights: {}".format(detection_weights_path))
        print("Path to object-tracking weights: {}".format(tracking_weights_path))
        print("Path to camera parameters: {}".format(camera_parameters_path))
        print("Path to output folder: {}".format(output_path))

    areArgumentsValid = evaluateArguments(input_path, detection_weights_path, tracking_weights_path, camera_parameters_path, output_path, verbose)
    isSoftwareValid = evaluateSoftware(verbose)

    if ((areArgumentsValid == True) and (isSoftwareValid == True)):
        excecuteSoftware(input_path, detection_weights_path, tracking_weights_path, camera_parameters_path, output_path, verbose)
    else:
        print("[ERROR] Aborting Software Execution!")
