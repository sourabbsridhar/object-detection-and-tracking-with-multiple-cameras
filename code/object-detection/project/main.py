import cv2
# from yolov4.tf import YOLOv4 # installed (global) library
# import tensorflow as tf
# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

from py_src.yolov4.tf import YOLOv4 # local library: to use the modified library!



'''Path to data'''
dataPath = 'data/test_1.jpg'
dataPathResized = 'data/test_1_resized.jpg'

''' Rescale images to 640 * 480'''
# This is not necessary because the yolov4 library also has this function
# img = cv2.imread(dataPath)
# width = 640
# height = 480
# dim = (width,height)
# resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
# print("Resized dimensions: ", resized.shape)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite(dataPathResized, resized)
''' End rescale'''


yolo = YOLOv4()
yolo.classes = "data/coco.names"
yolo.input_size = (640,480)

yolo.make_model()
yolo.load_weights("weights/yolov4.weights", weights_type = "yolo")

'''Infer image/video with IOU threshold = 0.5'''
yolo.inference(media_path="data/sample_vid_3_Trim.mp4", is_image=False,iou_threshold=0.5)
# yolo.inference(media_path="data/sample_vid_3.mkv", is_image=False,iou_threshold=0.5)
# yolo.inference(media_path=dataPath, is_image=False,iou_threshold=0.5)
# yolo.inference(media_path=dataPathResized, is_image=True, iou_threshold=0.5)