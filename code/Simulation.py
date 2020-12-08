import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import copy


def pflat(points):
    return np.divide(points, points[-1,:])


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


class ScenePoint:
    def __init__(self, id, position, velocity, acceleration_std=0):
        self.id = id
        self.position = np.array([position], dtype=np.float32).T
        self.velocity = np.array([velocity], dtype=np.float32).T
        self.acceleration_std = acceleration_std

    def update(self, deltaTime):
        self.velocity += deltaTime * np.random.uniform(0, self.acceleration_std, (3, 1))
        self.position += deltaTime * self.velocity


class ImagePoint:
    def __init__(self, scenePoint_id, camera_id, isVisible, position, velocity=np.zeros((2, 1))):
        self.scenePoint_id = scenePoint_id
        self.camera_id = camera_id
        self.position = position
        self.velocity = velocity
        self.isVisible = isVisible


def simulate_data(cameras, scenePoints, nrFrames, deltaTime):

    # Loop over time steps
    simulationData = list()
    for iTime in range(nrFrames):

        # Frame Data is all the data for this time instance, will be saved to Simulation Data
        frameData = dict()
        frameData['Scene Points'] = [copy.deepcopy(scenePoint) for scenePoint in scenePoints]
        frameData['Time'] = iTime * deltaTime

        # Calculate image points by projecting scene points onto cameras
        imagePoints = list()
        for scenePoint in scenePoints:
            for camera in cameras:
                projection, isVisible = camera.project(scenePoint)
                imagePoints.append(ImagePoint(scenePoint.id, camera.id, isVisible, projection))

        # If this isn't the first time instance, estimate the image point velocity
        if iTime is not 0:
            previousFrameData = simulationData[-1]
            for iPoint in range(len(imagePoints)):
                currentPos = imagePoints[iPoint].position
                prevPos = previousFrameData['Image Points'][iPoint].position
                velocity = (currentPos - prevPos) / deltaTime
                imagePoints[iPoint].velocity = velocity

        frameData['Image Points'] = [copy.deepcopy(imagePoint) for imagePoint in imagePoints]

        # Save frame data to simulation data
        simulationData.append(frameData)

        # Update scene point positions and velocities
        for scenePoint in scenePoints:
            scenePoint.update(deltaTime)

    return simulationData


# Parameters
nrFrames = 100
deltaTime = 0.1

# Define cameras
cameras = list()
cameras.append(Camera(id=len(cameras), R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                      t=[[0, 0, 2]],
                      K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
cameras.append(Camera(id=len(cameras), R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                      t=[[6, 0, 2]],
                      K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      t=[[-4, -2, 2]],
                      K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
nrCameras = len(cameras)

# Define scene points
scenePoints = list()
scenePoints.append(ScenePoint(id=len(scenePoints), position=[3, 3, 0], velocity=[-1, 0, 0]))
scenePoints.append(ScenePoint(id=len(scenePoints), position=[-3, -3, 0], velocity=[1, 0, 0]))
scenePoints.append(ScenePoint(id=len(scenePoints), position=[-2, 4, 0.5], velocity=[0.4, -1, 0]))
nrPoints = len(scenePoints)

# Get simulation data
simulationData = simulate_data(cameras, scenePoints, nrFrames, deltaTime)

# Detection
# Tracking

#data = Simulation.get_data(nrFrames)
# GT, each time step:  Camera frame: [detection = [camera views,id,...]]

# Fusion
# Kalman
# Reprojection


# ---------- Animations ----------
# Animation: 3D window
# Define animation updater
def animate_scene_points(iteration, simulationData, scatters):
    for iPoint, scenePoint in enumerate(simulationData[iteration]['Scene Points']):
        temp = (scenePoint.position[0], scenePoint.position[1], scenePoint.position[2])
        scatters[iPoint]._offsets3d = temp
    return scatters

# Initialize figure, plot cameras, initialize scatter plots
fig1 = plt.figure()
ax1 = p3.Axes3D(fig1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
for cam in cameras:
    cam.plotCamera(ax1)
scatters = [ax1.scatter(scenePoint.position[0], scenePoint.position[1], scenePoint.position[2]) for scenePoint in simulationData[0]['Scene Points']]

# Start animation
ani1 = animation.FuncAnimation(fig1, animate_scene_points, nrFrames, fargs=(simulationData, scatters), interval=50, blit=False, repeat=True)


# Animation: Camera views
# Define animation updater
def animate_image_points(iteration, simulationData, scatters):
    for iCam in range(len(scatters)):
        imagePoints = [imagePoint for imagePoint in simulationData[iteration]['Image Points'] if imagePoint.camera_id == iCam]
        for iPoint, imagePoint in enumerate(imagePoints):
            scatters[iCam][iPoint]._offsets = [[imagePoint.position[0].item(), imagePoint.position[1].item()]]

# Initialize plot. For each camera view, initialize scatter
fig2, axs = plt.subplots(len(cameras), 1)
scatters = list()
for iCam in range(len(cameras)):
    # Set axis range
    axs[iCam].set_xlim((0, 1920))
    axs[iCam].set_ylim((0, 1080))
    axs[iCam].invert_yaxis()
    axs[iCam].title.set_text('Camera {}'.format(cameras[iCam].id))

    # Initialize Scatters
    camScatters = [axs[iCam].scatter(imagePoint.position[0], imagePoint.position[1]) for imagePoint in simulationData[0]['Image Points'] if imagePoint.camera_id == iCam]
    scatters.append(camScatters)

# Start animation
ani2 = animation.FuncAnimation(fig2, animate_image_points, nrFrames, fargs=(simulationData, scatters), interval=50, blit=False, repeat=True)

plt.show()


