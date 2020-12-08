from __future__ import division, print_function, absolute_import

import argparse
import os
import time

import cv2
import numpy as np
import glob
import tensorflow as tf

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

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

def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "frames") # If frames is in folder img1

    # Create dictionary of all image filenames full paths
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    # if groundtruth folder name is gt
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # If detection file exist load the detection (what is in the files????)
    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)

  ##########################################################################

    # If groundtruth file exist load it
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
  ###########################################################################

    # if there are images in the dict image_filenames, take the next unprocessed image
    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE) # Read as grayscale
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys()) # Get the minimum frame number of images
        max_frame_idx = max(image_filenames.keys()) # Get the maximum frame number of images
    else:
        min_frame_idx = int(detections[:, 0].min()) # If no frames left take the min detection frame
        max_frame_idx = int(detections[:, 0].max()) # same for max

    info_filename = os.path.join(sequence_dir, "seqinfo.ini") # Get the sequence info file path
    # If it exist, save sequence info as a dictionary
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"]) # Set the update time proportional to FPS
    else:
        update_ms = None

    # Set the feature dimensions of the detections, now 137-10=127
    feature_dim = detections.shape[1] - 10 if detections is not None else 0

    # Define a dictionary with the specific
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx # Get all detections for the current frame

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    # Define variables
    DATASET_PATH = '/home/jonatan/SSY226/object-detection-and-tracking-with-multiple-cameras/dataset/stanford_drone/bookstore/'
    
    for seq in glob.glob(DATASET_PATH + '/*'):
        seq_info = gather_sequence_info(seq)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # TODO: Check the format of detections and rewrite it

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

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
min_confidence = 0.3
nms_max_overlap = 1.0
max_cosine_distance = 0.2
nn_budget = None


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#DETECTION_ROWS = 1000 # Only for debuging

# Main dataset path
PATH = "/home/jonatan/SSY226/object-detection-and-tracking-with-multiple-cameras/dataset/stanford_drone/bookstore/"

# Loop through all sequences (viedo0, video1 video2 etc.)
sequences = glob.glob(PATH + '/*')

for seq in sequences:
  seq = seq.split('/')[-1] # Only want the ending (video#)
  print("Initiate sequence: ", seq)

# Define all paths to be used
FRAME_PATH = PATH + seq+ '/frames/'
DET_PATH = PATH + seq + '/det/'
GT_PATH = PATH + seq + '/gt/'
OUTPUT_PATH = PATH + seq + '/deep_output/'

# Create output if it does not exist
if not os.path.exists(OUTPUT_PATH): 
  os.makedirs(OUTPUT_PATH)

seq_dets = np.loadtxt(DET_PATH + 'det.txt', delimiter=',') #, max_rows=DETECTION_ROWS)
#print(seq_dets)

# Create instance of the tracker
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric) # Initialize tracker with defined metric

# Define empty result for output
results = []
  

if not os.path.exists(OUTPUT_PATH + seq): 
  os.makedirs(OUTPUT_PATH + seq)

# Define the image encoder for feature extraction
image_encoder = ImageEncoder('mars-small128.pb') # Pre trained MARS weights
image_shape = image_encoder.image_shape
  
# Open output text file to save tracking in
with open(OUTPUT_PATH + seq + '/output.txt', 'w') as out_file:
  print("Number of frames = ", int(seq_dets[:,0].max())+1)
  # Loop over all frames in detection file
  for frame in range(int(seq_dets[:,0].max())+1): #detection and frame numbers begin at 1
    start_time = time.time() # Start timer for cycle
    dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
      
    #dets[:, 2:4] += dets[:, 0:2] #convert from [x1,y1,w,h] to [x1,y1,x2,y2]
    total_frames += 1

    img = cv2.imread(FRAME_PATH + '{}.jpg'.format(total_frames), cv2.IMREAD_COLOR)

    detection_list = []
    image_patch_batch = []
    patch_shape = image_shape[:2]
    for det in dets:
      x1, y1, x2, y2, confidence = det
      bbox = np.asarray([x1,y1,x2,y2])
      int_bbox = bbox.astype(np.int)
      # 
      if np.any(int_bbox[:2] >= int_bbox[2:]):
        feature = None
      else:
        img_patch = np.asarray(img.copy())
        image_patch = img_patch[int(y1):int(y2), int(x1):int(x2)]
        if image_patch is None:
          print("WARNING: Failed to extract image patch: %s." % str(box))
          image_patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
        else:
          image_patch = cv2.resize(image_patch, tuple(patch_shape[::-1]))

        if image_patch is not None:
          image_patch = np.reshape(image_patch, [-1, 128, 64, 3])
        #print(patch_shape[::-1])
        #print(np.shape(image_patch))
        
        feature = image_encoder(np.asarray(image_patch), 1)
        
      detection_list.append(Detection(bbox, confidence, feature))
    
    #print(detection_list)
    #break
    detections = [d for d in detection_list if d.confidence >= min_confidence] # Skip if less than 0.3

    boxes = np.array([d.to_tlwh for d in detections])
    #print("boxes", boxes)
    #break
    scores = np.array([d.confidence for d in detections])

    # Non maxima supression! Default to 1.0
    #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    #detections = [detections[i] for i in indices]

    # Store results.
    print('START TARCKER LOOP')
    for track in tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:
        print("SKIPPING track", track)
        continue
      print("DID NOT SKIP the track")
      bbox = track.tlbr()
      print(bbox)
      results.append([frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

      if(display):
        cv2.rectangle(im, (int(d[0]),int(d[3])), (int(d[2]),int(d[1])), (0,255,0), 1) # x1,y2,x2,y1
        text_with_backgroud(img, 'ID=%d'%d[4], (int(d[0]),int(d[3])), 
                                  font, scale=1, color_text=(0,0,0), color_bg=(0,255,0))
      
      if display:
        cv2.imshow('SDD',img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
          break
  
    # Update the tracker with the detections
    tracker.predict()
    tracker.update(detections)
    cycle_time = time.time() - start_time
    print(cycle_time)
    total_time += cycle_time


# Inform about calculation speeds
print(total_time)
print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

if(display):
  print("Note: to get real runtime results run without the option: --display")
