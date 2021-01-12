## Object Tracking
This branch contains the tracking part of the project. It is based on nwojke's two repos https://github.com/nwojke/deep_sort for deep sort and https://github.com/nwojke/cosine_metric_learning for the feature detection training.

## Setup and installation

First, clone the repository:
```
git clone https://git.chalmers.se/sourab/object-detection-and-tracking-with-multiple-cameras.git
```
Then clone the conda environment:
```
conda env create -f tracker-cpu.yml
```
Note that this might work for all platform, if not try to install dependencies on your own

## Usage
* 1. Go to cosine-metric folder, download a dataset to train the cosine-metric with
* 2. Train the cosine-metric and output a .pb file (trained model)
* 3. Go to deepsort folder. Copy .pb model here. 
* 4. Follow instructures in the following folder (README exist) to genereate the detections, download a dataset and finally run the DeepSORT to generate the tracking. 
* 5. genereate_videos.py can be used to vizualise the tracker.
