# Implementation of Object Detection Algorithm (To be provided by the object detection subgroup)

import cv2
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_input(input_path):

    print("In function display_input()")
    print("input_path: " + str(input_path))

    camera_1_images = str(input_path) + "\\**\\C1\\**\\*.png"
    camera_1_images = glob.glob(camera_1_images, recursive=True)
    camera_1_images = sorted(camera_1_images)

    camera_2_images = str(input_path) + "\\**\\C2\\**\\*.png"
    camera_2_images = glob.glob(camera_2_images, recursive=True)
    camera_2_images = sorted(camera_2_images)

    camera_3_images = str(input_path) + "\\**\\C3\\**\\*.png"
    camera_3_images = glob.glob(camera_3_images, recursive=True)
    camera_3_images = sorted(camera_3_images)

    camera_4_images = str(input_path) + "\\**\\C4\\**\\*.png"
    camera_4_images = glob.glob(camera_4_images, recursive=True)
    camera_4_images = sorted(camera_4_images)

    camera_5_images = str(input_path) + "\\**\\C5\\**\\*.png"
    camera_5_images = glob.glob(camera_5_images, recursive=True)
    camera_5_images = sorted(camera_5_images)

    camera_6_images = str(input_path) + "\\**\\C6\\**\\*.png"
    camera_6_images = glob.glob(camera_6_images, recursive=True)
    camera_6_images = sorted(camera_6_images)

    camera_7_images = str(input_path) + "\\**\\C7\\**\\*.png"
    camera_7_images = glob.glob(camera_7_images, recursive=True)
    camera_7_images = sorted(camera_7_images)

    assert (len(camera_1_images) == len(camera_2_images)) and (len(camera_2_images) == len(camera_3_images)) and (len(camera_3_images) == len(camera_4_images)) and \
        (len(camera_4_images) == len(camera_5_images)) and (len(camera_5_images) == len(camera_6_images)) and (len(camera_6_images) == len(camera_7_images)), "[ERROR]"

    for iterator in range(len(camera_1_images)):

        """
        print("camera_1_images[iterator] = " + camera_1_images[iterator])
        print("camera_2_images[iterator] = " + camera_2_images[iterator])
        print("camera_3_images[iterator] = " + camera_3_images[iterator])
        print("camera_4_images[iterator] = " + camera_4_images[iterator])
        print("camera_5_images[iterator] = " + camera_5_images[iterator])
        print("camera_6_images[iterator] = " + camera_6_images[iterator])
        print("camera_7_images[iterator] = " + camera_7_images[iterator])
        """

        img1 = mpimg.imread(camera_1_images[iterator])
        img2 = mpimg.imread(camera_2_images[iterator])
        img3 = mpimg.imread(camera_3_images[iterator])
        img4 = mpimg.imread(camera_4_images[iterator])
        img5 = mpimg.imread(camera_5_images[iterator])
        img6 = mpimg.imread(camera_6_images[iterator])
        img7 = mpimg.imread(camera_7_images[iterator])

        fig, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(img2)
        axs[0, 2].imshow(img3)
        axs[1, 1].imshow(img4)
        axs[2, 0].imshow(img5)
        axs[2, 1].imshow(img6)
        axs[2, 2].imshow(img7)
        plt.show()
        plt.close()
        

def object_detection(input_path):


    
    pass