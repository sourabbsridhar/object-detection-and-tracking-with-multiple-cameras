import numpy as np


def pflat(points):
    return np.divide(points, points[-1, :])


class Camera:
    def __init__(self, id, R, t, K, resolution=[1920, 1080]):
        self.id = id
        self.R = np.array(R)
        self.t = np.array(t).T
        self.K = np.array(K)
        self.resolution = resolution

        self.center = -np.dot(self.R.T, self.t)
        self.axis = np.dot(self.R.T, np.array([[0, 0, 1]]).T)

        self.P_norm = np.concatenate((self.R, self.t), axis=1)
        self.P = np.dot(self.K, self.P_norm)

    def project(self, scenePoint, normalized=False):
        # Project points
        equivalent_point = np.vstack([scenePoint.position,  1])
        if normalized:
            projectedPoint = pflat(np.dot(self.P_norm, equivalent_point))[:-1, :]
        else:
            projectedPoint = pflat(np.dot(self.P, equivalent_point))[:-1, :]

        # Round to nearest pixel
        projectedPoint = np.round(projectedPoint)

        # Check if projection is within camera image
        if 0 <= projectedPoint[0] <= self.resolution[0] and 0 <= projectedPoint[1] <= self.resolution[1]:
            isVisible = True
        else:
            isVisible = False
        return (projectedPoint, isVisible)

    def plotCamera(self, ax):

        ax.text(*self.center[0], *self.center[1], *self.center[2], 'Cam {}'.format(self.id))
        ax.quiver(self.center[0], self.center[1], self.center[2], self.axis[0], self.axis[1], self.axis[2], length=0.5)
        yaxis = np.dot(self.R.T, np.array([[0, 1, 0]]).T)
        ax.quiver(self.center[0], self.center[1], self.center[2], yaxis[0], yaxis[1], yaxis[2], length=0.5, color='r')


class ImagePoint:
    def __init__(self, detection_id, camera_id, detection_class, position):
        self.detection_id = detection_id
        self.camera_id = camera_id
        self.detection_class = detection_class
        self.position = position
        self.prev_position = None
        self.velocity = None


class Projection:
    def __init__(self, detection_id, camera_id, detection_class, position, velocity):
        self.detection_index = detection_id
        self.camera_index = camera_id
        self.detection_class = detection_class
        self.position = position
        self.velocity = velocity

    def __str__(self):
        return 'p: {}\tv: {}\tDetection: {}\tCamera: {}'.format(self.position, self.velocity, self.detection_index, self.camera_index)

    def __eq__(self, other):
        if type(other) != Projection:
            return False
        elif self.detection_index == other.detection_index and self.camera_index == other.camera_index:
            return True
        return False

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity


class Cluster:
    def __init__(self, index):
        self.projections = list()
        self.index = index
        self.position_average = None
        self.velocity_average = None

    def __eq__(self, other):
        if type(other) != Cluster:
            return False
        elif self.index != other.index:
            return False
        return True

    def add_projection(self, projection):
        self.projections.append(projection)
        self.update_averages()

    def validate_cluster(self, R, V):
        # Make sure projection isn't farther than R/2 and V/2 away from the average and that no
        for projection in self.projections:
            if np.linalg.norm(projection.get_position() - self.get_position()) > R/2 or \
                    np.linalg.norm(projection.get_velocity() - self.get_velocity()) > V/2:
                return False

        # Make sure no there are no projections from the same camera
        camera_list = [projection.camera_index for projection in self.projections]
        if len(camera_list) > len(set(camera_list)):
            return False

        return True

    def update_averages(self):
        self.position_average = np.mean([projection.position for projection in self.projections], axis=0)
        self.velocity_average = np.mean([projection.velocity for projection in self.projections], axis=0)

    def get_position(self):
        return self.position_average

    def get_velocity(self):
        return self.velocity_average

    def get_state(self):
        return np.vstack((self.position_average, self.velocity_average))


class OutputObject:
    def __init__(self, id, position, velocity):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.estimate_covariance = np.zeros((4, 4))
        self.observation_loss_count = 0
        self.observation_new_count = 1
        self.isValid = False

    def get_state(self):
        return np.vstack((self.position, self.velocity))

    def set_state(self, state):
        self.position = state[0:2]
        self.velocity = state[2:4]