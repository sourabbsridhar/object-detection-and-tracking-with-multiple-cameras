# Function and Class Definition for all Handover Methods

import numpy as np

def pflat(unnormalizedPoint):
    """
    Function to normalize projected points

    Parameters
    ----------
    unnormalizedPoint : Unnormalized Point

    Returns
    -------
    normalizedPoint : Normalized Point

    """
    return np.divide(unnormalizedPoint, unnormalizedPoint[-1, :])

class Camera:
    def __init__(self, id, R, t, K, resolution=[640,480]):
        """
        Function to initialize the objects of Camera class.

        Parameters
        ----------
        id        : Camera ID
        R         : Camera Rotation Matrix
        t         : Camera Translation Vector
        K         : Camera Intrinsic Parameters
        resolution: Camera Resolution (Default Resolution: [640, 480])

        Returns
        -------
        Camera : Initialized object of Camera Class

        """
        self.id = id
        self.R = np.array(R)
        self.t = np.array(t).T
        self.K = np.array(K)
        self.resolution = resolution

        self.center = -np.dot(self.R.T, self.t)
        self.axis = np.dot(self.R.T, np.array([0, 0, 1]).T)

        self.PNorm = np.concatenate((self.R, self.t), axis=1)
        self.P = np.dot(self.K, self.PNorm)        #TODO: To be checked with Alvin!!!!

    def project(self, scenePoint, normalized=False):
        """
        Function to project a scene point (3D) on to the image plane (2D) 

        Parameters
        ----------
        scenePoint : 3D coordinates of the scene point
        normalized : Flag to indicate if the camera matrix is normalized (Default Value: False)

        Returns
        -------
        projectedPoint : 2D coordinates of the projected point in the image plane
        isVisible      : Flag to indicate if the projected point is visible in the image plane

        """
        # Project Scene Points on to Image Plane
        homogeneousScenePoint = np.vstack([scenePoint.position, 1])
        if normalized:
            projectedPoint = pflat(np.dot(self.PNorm, homogeneousScenePoint))[:-1,:]
        else:
            projectedPoint = pflat(np.dot(self.P, homogeneousScenePoint))[:-1,:]

        # Round off projected point to nearest pixel
        projectedPoint = np.round(projectedPoint)

        # Check if the projected point is visible in the image plane
        if ((0 <= projectedPoint[0] <= self.resolution[0]) and (0 <= projectedPoint[1] <= self.resolution[1])):
            isVisible = True
        else:
            isVisible = False

        return (projectedPoint, isVisible)


    def plotCamera(self, ax):
        """
        Function to Plot Camera

        Parameters
        ----------
        ax : Axes to the plot

        Returns
        -------
        None

        """

        ax.text(*self.center[0], *self.center[1], *self.center[2], "Camera {}".format(self.id))
        ax.quiver(self.center[0], self.center[1], self.center[2], self.axis[0], self.axis[1], self.axis[2], length=0.5)
        yaxis = np.dot(self.R.T, np.array([0, 1, 0]).T)
        ax.quiver(self.center[0], self.center[1], self.center[2], yaxis[0], yaxis[1], yaxis[2], length=0.5, color='r')


class ImagePoint:
    def __init__(self, detectionId, cameraId, detectionClass, position):
        """
        Function to initialize the objects of ImagePoint class.

        Parameters
        ----------
        detectionId     : Detection ID
        cameraId        : Camera ID of the image point
        detectionClass  : Detection Class of the image point
        position        : Position of the image point

        Returns
        -------
        ImagePoint : Initialized object of ImagePoint Class

        """   
        self.detectionId = detectionId
        self.cameraId = cameraId
        self.detectionClass = detectionClass
        self.position = position
        self.previousPosition = None
        self.velocity = None


class Projection:
    def __init__(self, detectionId, cameraId, detectionClass, position, velocity):
        """
        Function to initialize the objects of Projection class

        Parameters
        ----------
        detectionId     : Detection ID
        cameraId        : Camera ID of the projection
        detectionClass  : Detection Class of the projection
        position        : Position of the projection
        velocity        : Velocity of the projection

        Returns
        -------
        Projection : Initialized object of Projection Class

        """  
        self.detectionIndex = detectionId
        self.cameraIndex = cameraId
        self.detectionClass = detectionClass
        self.position = position
        self.velocity = velocity

    def __str__(self):
        """
        Function to represent the object as a string

        Parameters
        ----------
        None

        Returns
        -------
        outputString : String representation of the object

        """
        return "p: {}\tv: {}\tdetection: {}\tcamera: {}".format(self.position, self.velocity, self.detectionIndex, self.cameraIndex)

    def __eq__(self, other):
        """
        Function to compare two objects of type Projection

        Parameters
        ----------
        None

        Returns
        -------
        comparisionResult : Boolean result of the comparision 

        """
        if (type(other) != Projection):
            comparisionResult = False
        elif ((self.detectionIndex == other.detectionIndex) and (self.cameraIndex == other.cameraIndex)):
            comparisionResult = True
        else:
            comparisionResult = False

        return comparisionResult

    def getPosition(self):
        """
        Function to get the position of the image point

        Parameters
        ----------
        None

        Returns
        -------
        position : Position of the image point

        """
        return self.position

    def getVelocity(self):
        """
        Function to get the velocity of the image point

        Parameters
        ----------
        None

        Returns
        -------
        velocity : Velocity of the image point

        """
        return self.velocity


class Cluster:
    def __init__(self, index):
        """
        Function to initialize the objects of Cluster class

        Parameters
        ----------
        index : Index of the cluster

        Returns
        -------
        Cluster : Initialized object of Cluster Class

        """
        self.projections = list()
        self.index = index
        self.positionAverage = None
        self.velocityAverage = None

    def __eq__(self, other):
        """
        Function to compare two objects of type Cluster

        Parameters
        ----------
        None

        Returns
        -------
        comparisionResult : Boolean result of the comparision 

        """
        if (type(other) != Cluster):
            comparisionResult = False
        elif (self.index == other.index):
            comparisionResult = True
        else:
            comparisionResult = False

        return comparisionResult

    def addProjection(self, projection):
        """
        Function to add projections to the cluster

        Parameters
        ----------
        projection : Projection to be added

        Returns
        -------
        None

        """
        self.projections.append(projection)
        self.updateAverages()

    def validateCluster(self, R, V):
        """
        Function to validate clusters

        Parameters
        ----------
        R : Distance Threshold
        V : Velocity Threshold

        Returns
        -------
        isClusterValid : Boolean Flag indicating the validity of the Cluster

        """
        isClusterValid = True

        for projection in self.projections:
            if (np.linalg.norm(projection.getPosition() - self.getPosition) > R/2) or \
                (np.linalg.norm(projection.getVelocity() - self.getVelocity) > V/2):
                isClusterValid = False

        cameraList = [projection.cameraIndex for projection in self.projections]
        if (len(cameraList) > len(set(cameraList))):
            isClusterValid = False

        return isClusterValid

    def updateAverages(self):
        """
        Function to update position and velocity averages of the cluster

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.positionAverage = np.mean([projection.position for projection in self.projections])
        self.velocityAverage = np.mean([projection.velocity for projection in self.projections])

    def getPosition(self):
        """
        Function to get the position of the cluster

        Parameters
        ----------
        None

        Returns
        -------
        position : Position of the cluster

        """
        return self.positionAverage

    def getVelocity(self):
        """
        Function to get the velocity of the cluster

        Parameters
        ----------
        None

        Returns
        -------
        velocity : Velocity of the cluster

        """
        return self.velocityAverage

    def getState(self):
        """
        Function to get the current state of the cluster
        The current state of the cluster is represented as s = [p, v]^T

        Parameters
        ----------
        None

        Returns
        -------
        state : State of the cluster represented as s = [p, v]^T

        """
        return np.vstack((self.positionAverage, self.velocityAverage))

class OutputObject:
    def __init__(self, id, position, velocity):
        """
        Function to initialize the objects of OutputObject class

        Parameters
        ----------
        id       : Index of the output object
        position : Position of the output object
        velocity : Velocity of the output object

        Returns
        -------
        OutputObject : Initialized object of OutputObject Class

        """
        self.id = id
        self.position = position
        self.velocity = velocity
        self.estimatedCovariance = np.zeros((4,4))
        self.observationLossCount = 0
        self.observationNewCount = 1
        self.isValid = False

    def getState(self):
        """
        Function to get the current state of the output object
        The current state of the output object is represented as s = [p, v]^T

        Parameters
        ----------
        None

        Returns
        -------
        state : State of the output object represented as s = [p, v]^T

        """
        return np.vstack((self.position, self.velocity))

    def setState(self, state):
        """
        Function to set the current state of the output object
        The current state of the output object is represented as s = [p, v]^T

        Parameters
        ----------
        state : User defined state of the output object

        Returns
        -------
        None

        """        
        self.position = state[0:2]
        self.velocity = state[2:4]
