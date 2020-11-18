
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import tensorflow as tf
from yolov4.tf import YOLOv4
import cv2

# ''' Add data and weights directories to path '''
# pathWeight = "weights"
# pathData = "data"
# os.environ['PATH'] += ':'+pathData
# os.environ['PATH'] += ':'+pathWeight
for p in sys.path:
    print(p)

dataPath = 'data/test_1.jpg'
dataPathResized = 'data/test_1_resized.jpg'
''' Rescale images to 640 * 480'''
img = cv2.imread(dataPath)
width = 640
height = 480
dim = (width,height)
resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
# print("Resized dimensions: ", resized.shape)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(dataPathResized, resized)

yolo = YOLOv4()
yolo.classes = "data/coco.names"
yolo.input_size = (640,480)

yolo.make_model()
yolo.load_weights("weights/yolov4.weights", weights_type = "yolo")

# yolo.inference(media_path=dataPath)
# yolo.inference(media_path="data/sample_vid_3_Trim.mp4", is_image=False,iou_threshold=0.5)
yolo.inference(media_path=dataPath, is_image=False,iou_threshold=0.5)

