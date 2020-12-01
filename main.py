# Implementation of all overall software

import argparse

parser = argparse.ArgumentParser(description="Multi-object Detection and Tracking using Multiple Cameras")
parser.add_argument("-i", "--input", help="path to input folder")
parser.add_argument("-w", "--weights", help="path to object-detection weights")
parser.add_argument("-cp", "--cameraparameters", help="path to camera parameters")
parser.add_argument("-o", "--output", help="path to output folder")
parser.add_argument("-v", "--verbose", help="display log", action="store_true")
parser.parse_args()