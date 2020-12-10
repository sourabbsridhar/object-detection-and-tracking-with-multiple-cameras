import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import copy
from Handover_Library import Camera, ImagePoint
from Handover_Fusion import fusion
from Handover_Tracking import output_object_tracking
from project_2d_position import ground_projections


class ScenePoint:
    def __init__(self, id, position, velocity, acceleration_std=0):
        self.id = id
        self.position = np.array([position], dtype=np.float32).T
        self.velocity = np.array([velocity], dtype=np.float32).T
        self.acceleration_std = acceleration_std

    def update(self, deltaTime):
        self.velocity += deltaTime * np.vstack((np.random.normal(0, self.acceleration_std, (2, 1)), 0))
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

        # Calculate image points by projecting scene points onto cameras, if it is visible, save it
        imagePoints = list()
        for scenePoint in scenePoints:
            for camera in cameras:
                projection, isVisible = camera.project(scenePoint)
                if isVisible:
                    imagePoints.append(ImagePoint(scenePoint.id, camera.id, 0, projection))

        # If image point was visible in previous time instance, get previous position and calculate velocity
        if iTime is not 0:
            previousFrameData = simulationData['Frame Data'][-1]

            for imagePoint in imagePoints:
                prevPointList = [ip for ip in previousFrameData['Image Points']
                             if ip.detection_id == imagePoint.detection_id and ip.camera_id == imagePoint.camera_id]
                if len(prevPointList) > 0:
                    prevPoint = prevPointList[0]
                    imagePoint.prev_position = prevPoint.position
                    imagePoint.velocity = (imagePoint.position - prevPoint.position) / deltaTime


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
        nrFrames = 200
        deltaTime = 0.1

        # Define cameras
        cameras = list()
        cameras.append(Camera(id=len(cameras), R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                              t=[[0, 0, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                              t=[[6, 2, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              t=[[-4, -2, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                              t=[[3, -5, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        nrCameras = len(cameras)

        # Define scene points
        scenePoints = list()
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[3, 3, 1.5], velocity=[-0.5, -0.5, 0]))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[4, -4, 0.5], velocity=[-0.6, 0.6, 0]))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[-10, 0, 1], velocity=[1, 0, 0]))

        return simulate_data(cameras, scenePoints, nrFrames, deltaTime)

    else:
        raise Exception('There is no scenario {}. Please insert a valid scenario number.'.format(scenario_number))


# Get simulation data
simulationData = simulation_scenario(1)

# Fusion Parameters
R = 6
V = 1

# Tracking parameters
deltaTime = 0.1
distance_maximum = 5
observation_loss_maximum = 3
observation_new_minimum = 5

# Handover Loop
output_objects = list()
handover_simulation_data = list()
for iTime in range(simulationData['Nr Frames']):

    # Perform Handover
    imagePoints = simulationData['Frame Data'][iTime]['Image Points']
    projections = ground_projections(imagePoints, simulationData['Cameras'], 0, deltaTime)
    clusters = fusion(projections, R, V)
    output_objects = output_object_tracking(output_objects, clusters, deltaTime, distance_maximum, observation_loss_maximum, observation_new_minimum)
    validated_output_objects = [obj for obj in output_objects if obj.isValid]


    # Save frame data and append to handover simulation data
    frame_data = dict()
    frame_data['Projections'] = copy.deepcopy(projections)
    frame_data['Clusters'] = copy.deepcopy(clusters)
    frame_data['Output Objects'] = copy.deepcopy(output_objects)
    frame_data['Valid Output Objects'] = copy.deepcopy(validated_output_objects)
    handover_simulation_data.append(frame_data)


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


# Animation: Handover points
def animate_handover_points(iteration, sim_data, handover_sim_data, scatters):
    if len(sim_data['Frame Data'][iteration]['Scene Points']) > 0:
        scatters[0]._offsets = np.vstack([sp.position[0:2].T for sp in sim_data['Frame Data'][iteration]['Scene Points']])
    if len(handover_sim_data[iteration]['Projections']) > 0:
        scatters[1]._offsets = np.vstack([pr.position.T for pr in handover_sim_data[iteration]['Projections']])
    if len(handover_sim_data[iteration]['Clusters']) > 0:
        scatters[2]._offsets = np.vstack([cl.get_position().T for cl in handover_sim_data[iteration]['Clusters']])
    if len(handover_sim_data[iteration]['Valid Output Objects']) > 0:
        scatters[3]._offsets = np.vstack([ob.position.T for ob in handover_sim_data[iteration]['Valid Output Objects']])
    if len(sim_data['Frame Data'][iteration]['Scene Points']) > 0:
        scatters[4]._offsets = np.vstack([sp.velocity[0:2].T for sp in sim_data['Frame Data'][iteration]['Scene Points']])
    if len(handover_sim_data[iteration]['Projections']) > 0:
        scatters[5]._offsets = np.vstack([pr.velocity.T for pr in handover_sim_data[iteration]['Projections']])
    if len(handover_sim_data[iteration]['Clusters']) > 0:
        scatters[6]._offsets = np.vstack([cl.get_velocity().T for cl in handover_sim_data[iteration]['Clusters']])
    if len(handover_sim_data[iteration]['Valid Output Objects']) > 0:
        scatters[7]._offsets = np.vstack([ob.velocity.T for ob in handover_sim_data[iteration]['Valid Output Objects']])


# Initialize plot
fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6))
ax3[0].set_xlim((-15, 15))
ax3[0].set_ylim((-15, 15))
ax3[1].set_xlim((-2, 2))
ax3[1].set_ylim((-2, 2))

# Initialize scatters for scene points, projections, clusters and objects
scatters = list()
scatters.append(ax3[0].scatter([sp.position[0] for sp in simulationData['Frame Data'][0]['Scene Points']],
                            [sp.position[1] for sp in simulationData['Frame Data'][1]['Scene Points']],
                            color='k', alpha=0.8))
scatters.append(ax3[0].scatter([pr.position[0] for pr in handover_simulation_data[0]['Projections']],
                            [pr.position[1] for pr in handover_simulation_data[0]['Projections']],
                            marker='x', alpha=1))
scatters.append(ax3[0].scatter([cl.get_position()[0] for cl in handover_simulation_data[0]['Clusters']],
                            [cl.get_position()[1] for cl in handover_simulation_data[0]['Clusters']],
                            marker='o', alpha=1, s=200, facecolors='none', edgecolor='orange'))
scatters.append(ax3[0].scatter([ob.position[0] for ob in handover_simulation_data[0]['Valid Output Objects']],
                            [ob.position[1] for ob in handover_simulation_data[0]['Valid Output Objects']],
                            alpha=1))
scatters.append(ax3[1].scatter([sp.velocity[0] for sp in simulationData['Frame Data'][0]['Scene Points']],
                            [sp.velocity[1] for sp in simulationData['Frame Data'][1]['Scene Points']],
                            color='k', alpha=0.8))
scatters.append(ax3[1].scatter([pr.velocity[0] for pr in handover_simulation_data[0]['Projections']],
                            [pr.velocity[1] for pr in handover_simulation_data[0]['Projections']],
                            marker='x', alpha=1))
scatters.append(ax3[1].scatter([cl.get_velocity()[0] for cl in handover_simulation_data[0]['Clusters']],
                            [cl.get_velocity()[1] for cl in handover_simulation_data[0]['Clusters']],
                            marker='o', alpha=1, s=200, facecolors='none', edgecolor='orange'))
scatters.append(ax3[1].scatter([ob.velocity[0] for ob in handover_simulation_data[0]['Valid Output Objects']],
                            [ob.velocity[1] for ob in handover_simulation_data[0]['Valid Output Objects']],
                            alpha=1))

ax3[0].title.set_text('Position')
ax3[1].title.set_text('Velocity')
ax3[0].legend(['True Points', 'Projections', 'Clusters', 'Validated Objects'])
ax3[1].legend(['True Points', 'Projections', 'Clusters', 'Validated Objects'])

# Start animation
ani3 = animation.FuncAnimation(fig3, animate_handover_points, simulationData['Nr Frames'], fargs=(simulationData, handover_simulation_data, scatters), interval=50, blit=False, repeat=True)

#writervideo = animation.FFMpegWriter(fps=15)
#ani3.save('Handover_Simulation.mp4', writer=writervideo)

plt.show()


