import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


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

    def project(self, points, normalized=False):

        # Project points
        equivalent_points = np.vstack([points,  np.ones((1, points.shape[1]))])
        if normalized:
            projectedPoints = pflat(np.dot(self.P_norm, equivalent_points))[:-1, :]
        else:
            projectedPoints = pflat(np.dot(self.P, equivalent_points))[:-1, :]

        # Round to nearest pixel
        projectedPoints = np.round(projectedPoints)

        # Check if projection is within camera image
        inImage = list()
        for iPoint in range(points.shape[1]):
            if 0 <= projectedPoints[0, iPoint] <= self.resolution[0] and 0 <= projectedPoints[1, iPoint] <= self.resolution[1]:
                inImage.append(True)
            else:
                inImage.append(False)
        return (projectedPoints, inImage)

    def plotCamera(self, ax):

        ax.text(*self.center[0], *self.center[1], *self.center[2], 'Cam {}'.format(self.id))
        ax.quiver(self.center[0], self.center[1], self.center[2], self.axis[0], self.axis[1], self.axis[2], length=0.5)
        yaxis = np.dot(self.R.T, np.array([[0, 1, 0]]).T)
        ax.quiver(self.center[0], self.center[1], self.center[2], yaxis[0], yaxis[1], yaxis[2], length=0.5, color='r')


# Parameters
nrFrames = 500

# TODO add 2 more cameras
# Create cameras
cameras = list()
cameras.append(Camera(id=1, R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                      t=[[0, 0, 2]],
                      K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
cameras.append(Camera(id=2, R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                      t=[[6, 0, 2]],
                      K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))


# Create point 3D locations over time
pointData = [np.array([[3.0, -3.0, -2], [3.0, -3.0, 4], [0.0, 0.0, 0.5]])]
pointVelocities = np.array([[-0.1, 0.1, 0.03], [0, 0, -0.1], [0, 0, 0]])
for i in range(nrFrames):
    pointData.append(pointData[-1] + pointVelocities)
nrPoints = pointData[0].shape[1]

# Create point projections in cameras over time
pointProjections = list()
pointInImage = list()
for camera in cameras:
    cameraProjections = list()
    cameraInImage = list()
    for data in pointData:
        projections, inImage = camera.project(data)
        cameraProjections.append(projections)
        cameraInImage.append(inImage)
    pointProjections.append(cameraProjections)
    pointInImage.append(cameraInImage)


# Detection
# Tracking

data = Simulation.get_data(nrFrames)
# GT, each time step:  Camera frame: [detection = [camera views,id,...]]

# Fusion
# Kalman
# Reprojection




# ----- Plot 3D space -----
def animate_3D_scatters(iteration, data, scatters):
    for iPoint in range(data[0].shape[1]):
        temp = (data[iteration][0:1, iPoint], data[iteration][1:2, iPoint], data[iteration][2:3, iPoint])
        scatters[iPoint]._offsets3d = temp
    return scatters


fig1 = plt.figure()
ax1 = p3.Axes3D(fig1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

for cam in cameras:
    cam.plotCamera(ax1)


# Initialize scatter plot for each point
scatters = [ax1.scatter(pointData[0][0, i], pointData[0][1, i], pointData[0][2, i]) for i in range(nrPoints)]

# Start animation
ani1 = animation.FuncAnimation(fig1, animate_3D_scatters, nrFrames, fargs=(pointData, scatters), interval=50, blit=False, repeat=True)


# ----- Plot camera views -----
def animate_camera_scatters(iteration, data, scatters):
    for iCam in range(len(scatters)):
        for iPoint in range(len(scatters[iCam])):
            temp = [[data[iCam][iteration][0, iPoint], data[iCam][iteration][1, iPoint]]]
            scatters[iCam][iPoint]._offsets = temp

fig2, axs = plt.subplots(len(cameras), 1)
scatters = list()
for iCam in range(len(cameras)):
    # Set axis range
    axs[iCam].set_xlim((0, 1920))
    axs[iCam].set_ylim((0, 1080))
    axs[iCam].invert_yaxis()
    axs[iCam].title.set_text('Camera {}'.format(cameras[iCam].id))

    # Initialize Scatters
    camScatters = [axs[iCam].scatter(pointProjections[iCam][0][0, iPoint], pointProjections[iCam][0][1, iPoint]) for iPoint in range(nrPoints)]
    scatters.append(camScatters)

ani2 = animation.FuncAnimation(fig2, animate_camera_scatters, nrFrames, fargs=(pointProjections, scatters), interval=50, blit=False, repeat=True)

plt.show()


