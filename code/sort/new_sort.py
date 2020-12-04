import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import scipy.linalg

import cv2
import glob
import time
import argparse
#from filterpy.kalman import KalmanFilter

# Deepsort functions
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# Comparing metric
import motmetrics as mm

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  #if h == 0:
  #  return np.array([0,0,0,0]).reshape((4,1))
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

##### REPLACE KALMAN ##########


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # REWRITE THE KALMAN EQUATIONS

    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10. # 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict() # KalmanFilter from filterpy.kalman
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

##############################################


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    # Put detection and tracks in respective list depending on if it could be match given the criterias
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections, m = (det_index, track_index)
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :]) # Update Track(m[1]) with detection[m[0]]

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i,:])
      self.trackers.append(trk) # Start new tracking of the detection that did not find any match with currnt trackings

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


def text_with_backgroud(img, text, org, font, scale, color_text=(0,0,0), color_bg=(255,255,255)):
    (txt_w, txt_h) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
    cv2.rectangle(img, tuple([org[0], org[1]-txt_h-3]), tuple([org[0]+txt_w, org[1]]), color_bg, cv2.FILLED)
    cv2.putText(img, text, tuple([org[0],org[1]-3]), font, scale, color_text)

########################################################################
############################# MAIN #####################################
########################################################################

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

  # Create instance of the SORT tracker
  mot_tracker = Sort(max_age=90,  # "Maximum number of frames to keep alive a track without associated detections."
                      min_hits=0, # Minimum number of associated detections before track is initialised
                      iou_threshold=0.3) # Minimum IOU for match
  

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
