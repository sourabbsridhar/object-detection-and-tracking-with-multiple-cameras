import os
import numpy as np

from skimage import io
import scipy.linalg

import cv2
import glob
import time
#from filterpy.kalman import KalmanFilter

# Deepsort functions
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# Comparing metric
import motmetrics as mm

#####################################################################
# FROM GENERATE DETECTIONS ##########################################
#####################################################################
class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


#################### MAIN ############################################

display = True
total_time = 0.0
total_frames = 0
colours = np.random.rand(32, 3) #used only for display
#DETECTION_ROWS = 1000 # Only for debuging

# Main dataset path
PATH = "/home/jonatan/SSY226/object-detection-and-tracking-with-multiple-cameras/dataset/stanford_drone/bookstore/"

# Define all paths to be used
FRAME_PATH = PATH + 'frames/'
DET_PATH = PATH + 'det/'
GT_PATH = PATH + 'gt/'
OUTPUT_PATH = PATH + 'output/'

# Create output if it does not exist
if not os.path.exists(OUTPUT_PATH): 
  os.makedirs(OUTPUT_PATH)


# Loop through all sequences (viedo0, video1 video2 etc.)
sequences = glob.glob(DET_PATH + '/*')

for seq in sequences:
  seq = seq.split('/')[-1] # Only want the ending (video#)
  print("Initiate sequence: ", seq)

  seq_dets = np.loadtxt(DET_PATH + seq + '/generated_detections.txt', delimiter=',') #, max_rows=DETECTION_ROWS)
  #print(seq_dets)

  # Create instance of the DEEP SORT tracker
  metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # Metric for cost matrix
  # Initiate the tracker
  tracker = Tracker(metric)

  # Object detection group do non-max-supression?

  if not os.path.exists(OUTPUT_PATH + seq): 
    os.makedirs(OUTPUT_PATH + seq)
  
  # Open output text file to save tracking in
  with open(OUTPUT_PATH + seq + '/output.txt', 'w') as out_file:
    print("Number of frames = ", int(seq_dets[:,0].max())+1)
    # Loop over all frames in detection file
    for frame in range(int(seq_dets[:,0].max())+1): #detection and frame numbers begin at 1
      dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
      #dets[:, 2:4] += dets[:, 0:2] #convert from [x1,y1,w,h] to [x1,y1,x2,y2]
      total_frames += 1
      #print(dets)

      # Update the tracker with the detections
      start_time = time.time()
      trackers = mot_tracker.update(dets)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      if(display):
        im = cv2.imread(FRAME_PATH + seq + '/%d.jpg'%(frame+1))
        #cv2.imshow("Frame: %d", im)



      # Write out the tracking result for the current frame
      # [frame, ID, x1, y1, x2, y2, class, -1, -1, -1]
      for d in trackers:
        # Write frame, ID, x1, y1, x2, y2
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2],d[3]),file=out_file)
        #print("\nBBOX",d[0],d[1],d[2],d[3], '\n')
        #break
        # Display the tracking bounding boxes
        if(display):
          font = cv2.FONT_HERSHEY_COMPLEX_SMALL
          cv2.rectangle(im, (int(d[0]),int(d[3])), (int(d[2]),int(d[1])), (0,255,0), 1) # x1,y2,x2,y1
          text_with_backgroud(im, 'ID=%d'%d[4], (int(d[0]),int(d[3])), 
                                  font, scale=1, 
                                  color_text=(0,0,0), color_bg=(0,255,0))

          cv2.rectangle(im, tuple([1,30]), tuple([10,40]), (0,255,0), 1)  
          cv2.putText(im, ': detected', tuple([12,40]), font, 1, (0,255,0))
      
      if display:
        cv2.imshow('SDD',im)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
          break

# Inform about calculation speeds
print(total_time)
print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

if(display):
  print("Note: to get real runtime results run without the option: --display")


