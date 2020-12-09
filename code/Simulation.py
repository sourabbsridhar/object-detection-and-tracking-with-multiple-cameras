import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import copy
from Handover_Library import Camera, ImagePoint
from Handover_Fusion import fusion
from project_2d_position import ground_projections


class ScenePoint:
    def __init__(self, id, position, velocity, acceleration_std=0):
        self.id = id
        self.position = np.array([position], dtype=np.float32).T
        self.velocity = np.array([velocity], dtype=np.float32).T
        self.acceleration_std = acceleration_std

    def update(self, deltaTime):
        self.velocity += deltaTime * np.random.uniform(0, self.acceleration_std, (3, 1))
        self.position += deltaTime * self.velocity


def simulate_data(cameras, scenePoints, nrFrames, deltaTime):

    # Loop over time steps
    simulationData = dict()
    simulationData['Cameras'] = cameras
    simulationData['Nr Frames'] = nrFrames
    simulationData['Delta Time'] = deltaTime
    simulationData['Frame Data'] = list()

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
                imagePoints.append(ImagePoint(scenePoint.id, camera.id, 0, projection, isVisible=isVisible))

        # If this isn't the first time instance, estimate the image point velocity
        if iTime is not 0:
            previousFrameData = simulationData['Frame Data'][-1]
            for iPoint in range(len(imagePoints)):
                currentPos = imagePoints[iPoint].position
                prevPos = previousFrameData['Image Points'][iPoint].position
                velocity = (currentPos - prevPos) / deltaTime
                imagePoints[iPoint].velocity = velocity

        frameData['Image Points'] = [copy.deepcopy(imagePoint) for imagePoint in imagePoints]

        # Save frame data to simulation data
        simulationData['Frame Data'].append(frameData)

        # Update scene point positions and velocities
        for scenePoint in scenePoints:
            scenePoint.update(deltaTime)

    return simulationData


def simulation_scenario(scenario_number):

    if scenario_number is 1:

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
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[0, 0, 0.5], velocity=[0.4, -1, 0]))
        nrPoints = len(scenePoints)

        return simulate_data(cameras, scenePoints, nrFrames, deltaTime)

    else:
        raise Exception('There is no scenario {}. Please insert a valid scenario number.'.format(scenario_number))


# Get simulation data
simulationData = simulation_scenario(1)

# Parameters
# Fusion
R = 1
V = 1

# Handover Loop
for iTime in range(simulationData['Nr Frames']):

    # Perform Handover
    imagePoints = [imagePoint for imagePoint in simulationData['Frame Data'][iTime]['Image Points'] if imagePoint.isVisible]
    projections = ground_projections(imagePoints, simulationData['Cameras'], 0)
    clusters = fusion(projections, R, V)
    # output_objects =



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
    for iPoint, scenePoint in enumerate(simulationData['Frame Data'][iteration]['Scene Points']):
        temp = (scenePoint.position[0], scenePoint.position[1], scenePoint.position[2])
        scatters[iPoint]._offsets3d = temp
    return scatters

# Initialize figure, plot cameras, initialize scatter plots
fig1 = plt.figure()
ax1 = p3.Axes3D(fig1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
for cam in simulationData['Cameras']:
    cam.plotCamera(ax1)
scatters = [ax1.scatter(scenePoint.position[0], scenePoint.position[1], scenePoint.position[2]) for scenePoint in simulationData['Frame Data'][0]['Scene Points']]

# Start animation
ani1 = animation.FuncAnimation(fig1, animate_scene_points, simulationData['Nr Frames'], fargs=(simulationData, scatters), interval=50, blit=False, repeat=True)


# Animation: Camera views
# Define animation updater
def animate_image_points(iteration, simulationData, scatters):
    for iCam in range(len(scatters)):
        imagePoints = [imagePoint for imagePoint in simulationData['Frame Data'][iteration]['Image Points'] if imagePoint.camera_id == iCam]
        for iPoint, imagePoint in enumerate(imagePoints):
            scatters[iCam][iPoint]._offsets = [[imagePoint.position[0].item(), imagePoint.position[1].item()]]

# Initialize plot. For each camera view, initialize scatter
fig2, axs = plt.subplots(len(simulationData['Cameras']), 1)
scatters = list()
for iCam in range(len(simulationData['Cameras'])):
    # Set axis range
    axs[iCam].set_xlim((0, 1920))
    axs[iCam].set_ylim((0, 1080))
    axs[iCam].invert_yaxis()
    axs[iCam].title.set_text('Camera {}'.format(simulationData['Cameras'][iCam].id))

    # Initialize Scatters
    camScatters = [axs[iCam].scatter(imagePoint.position[0], imagePoint.position[1]) for imagePoint in simulationData['Frame Data'][0]['Image Points'] if imagePoint.camera_id == iCam]
    scatters.append(camScatters)

# Start animation
ani2 = animation.FuncAnimation(fig2, animate_image_points, simulationData['Nr Frames'], fargs=(simulationData, scatters), interval=50, blit=False, repeat=True)

plt.show()


