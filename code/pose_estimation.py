# Implementation of the Pose Estimation Function (To be provided by the handover subgroup)

import numpy as np

class PoseEstimation:
    def __init__(self, mode):
        self.mode = mode

    def OfflinePoseEstimation(self):
        pass

    def OnlinePoseEstimation(self):
        pass

    def EstimatePose(self):

        if(self.mode == "online"):
            self.OnlinePoseEstimation()
        else:
            self.OnlinePoseEstimation()

    