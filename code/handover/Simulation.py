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
    def __init__(self, id, position, velocity, acceleration_std=None, rotation_period=None):
        self.id = id
        self.position = np.array([position], dtype=np.float32).T
        self.velocity = np.array([velocity], dtype=np.float32).T
        self.acceleration_std = acceleration_std
        self.rotation_period = rotation_period

    def update(self, deltaTime, time=None):

        if self.acceleration_std is not None:
            self.velocity += deltaTime * np.vstack((np.random.normal(0, self.acceleration_std, (2, 1)), 0))
        if self.rotation_period is not None and time is not None:
            self.velocity[0] = self.velocity[0] * np.cos(self.rotation_period * deltaTime) \
                               - self.velocity[1] * np.sin(self.rotation_period * deltaTime)
            self.velocity[1] = self.velocity[0] * np.sin(self.rotation_period * deltaTime) \
                               + self.velocity[1] * np.cos(self.rotation_period * deltaTime)

        self.position += deltaTime * self.velocity


def simulate_data(cameras, scenePoints, nrFrames, deltaTime, image_point_noise_std):

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

                # Add noise to the projection
                projection += np.round(np.random.normal(0, image_point_noise_std, (2, 1)))

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
            scenePoint.update(deltaTime, iTime*deltaTime)

    # Remove all image points that do not have a previous position
    for frame in simulationData['Frame Data']:
        frame['Image Points'] = [ip for ip in frame['Image Points'] if ip.prev_position is not None]

    return simulationData


def simulation_scenario(scenario_number):

    if scenario_number is 1:
        # Parameters
        nrFrames = 300
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
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[3, 3, 1.5], velocity=[-0.5, -0.5, 0], acceleration_std=0.1))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[4, -4, 0.5], velocity=[-0.6, 0.6, 0], acceleration_std=0.2))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[-10, 0, 1], velocity=[1, 0, 0], acceleration_std=0.3))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[2, -10, 1], velocity=[0.1, 0.7, 0], acceleration_std=0.2))

        return simulate_data(cameras, scenePoints, nrFrames, deltaTime)

    elif scenario_number is 2:
        # Parameters
        nrFrames = 350
        deltaTime = 0.1

        # Define cameras
        cameras = list()
        cameras.append(Camera(id=len(cameras), R=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                              t=[[0, 0, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                              t=[[6, 2, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                              t=[[-4, -2, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        cameras.append(Camera(id=len(cameras), R=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                              t=[[3, -5, 4]],
                              K=[[400, 0, 960], [0, 400, 540], [0, 0, 1]]))
        nrCameras = len(cameras)

        # Define scene points
        scenePoints = list()
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[5, 5, 1.4], velocity=[-0.5, -0.5, 0],
                                      acceleration_std=0, rotation_period=0.1))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[-5.2, 5, 0.5], velocity=[0.5, -0.5, 0],
                                      acceleration_std=0, rotation_period=-0.05))
        scenePoints.append(ScenePoint(id=len(scenePoints), position=[7, 0, 1.1], velocity=[-0.5, -0.3, 0],
                                      acceleration_std=0, rotation_period=0))

        return simulate_data(cameras, scenePoints, nrFrames, deltaTime)

    elif scenario_number is 3:
        # Parameters
        nrFrames = 300
        deltaTime = 0.1
        imagePoint_noise_STD = 5
        nrScenePoints = 3

        # Define cameras
        cameras = list()
        cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              t=[[5.8, 4, 4]],
                              K=[[150, 0, 320], [0, 150, 240], [0, 0, 1]],
                              resolution=(640, 480)))
        cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              t=[[-5.8, 4, 4]],
                              K=[[150, 0, 320], [0, 150, 240], [0, 0, 1]],
                              resolution=(640, 480)))
        cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              t=[[5.8, -4, 4]],
                              K=[[150, 0, 320], [0, 150, 240], [0, 0, 1]],
                              resolution=(640, 480)))
        cameras.append(Camera(id=len(cameras), R=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              t=[[-5.8, -4, 4]],
                              K=[[150, 0, 320], [0, 150, 240], [0, 0, 1]],
                              resolution=(640, 480)))
        nrCameras = len(cameras)

        # Define scene points
        scenePoints = list()
        for i in range(nrScenePoints):
            height = np.max((np.random.normal(1.8, 0.4), 0)) / 2
            center_distance = np.random.normal(20, 1)
            center_angle = np.random.uniform(0, 2 * np.pi)
            center = np.array([0, 0, 0])
            startPos = center + np.array([center_distance * np.cos(center_angle), center_distance * np.sin(center_angle), height])

            veldirection = center_angle + np.pi + np.random.normal(0, 0.3)
            speed = np.random.normal(1.2, 0.3)
            startVel = np.array([speed*np.cos(veldirection), speed*np.sin(veldirection), 0])

            accelerationSTD = np.max((np.random.normal(0.3, 0.1), 0.01))
            randomPeriod = np.random.normal(0, 0.025)

            scenePoints.append(ScenePoint(id=len(scenePoints), position=startPos, velocity=startVel,
                                          acceleration_std=accelerationSTD, rotation_period=randomPeriod))

        return simulate_data(cameras, scenePoints, nrFrames, deltaTime, imagePoint_noise_STD)

    else:
        raise Exception('There is no scenario {}. Please insert a valid scenario number.'.format(scenario_number))


# Get simulation data
simulationData = simulation_scenario(3)

# Projection Parameters
ground_height = 1
maximum_speed = 4

# Fusion Parameters
R = 2.5
V = 6

# Tracking parameters
deltaTime = 0.1
distance_maximum = 2
observation_new_minimum = 3
observation_loss_maximum = 3

# Handover Loop
output_objects = list()
handover_simulation_data = list()
output_objectCount = 0  # For setting IDs of new objects to the next unique number
for iTime in range(simulationData['Nr Frames']):
    # Perform Handover
    imagePoints = simulationData['Frame Data'][iTime]['Image Points']
    projections = ground_projections(imagePoints, simulationData['Cameras'], ground_height, deltaTime)
    clusters = fusion(projections, R, V)
    (output_objects, output_objectCount) = output_object_tracking(output_objects, clusters, output_objectCount, deltaTime, distance_maximum, observation_loss_maximum, observation_new_minimum)
    validated_output_objects = [obj for obj in output_objects if obj.isValid]

    # Save frame data and append to handover simulation data
    frame_data = dict()
    frame_data['Projections'] = copy.deepcopy(projections)
    frame_data['Clusters'] = copy.deepcopy(clusters)
    frame_data['Output Objects'] = copy.deepcopy(output_objects)
    frame_data['Valid Output Objects'] = copy.deepcopy(validated_output_objects)
    handover_simulation_data.append(frame_data)


# Evaluation Metrics


# Plotting
def plot_camera_groundview(cameras, ground_height, ax):

    for camera in cameras:
        imagePoints = list()
        imagePoints.append(ImagePoint(0, camera.id, 0, np.array([[0], [0]])))
        imagePoints.append(ImagePoint(0, camera.id, 0, np.array([[camera.resolution[0]], [0]])))
        imagePoints.append(ImagePoint(0, camera.id, 0, np.array([[camera.resolution[0]], [camera.resolution[1]]])))
        imagePoints.append(ImagePoint(0, camera.id, 0, np.array([[0], [camera.resolution[1]]])))

        groundProjections = ground_projections(imagePoints, [camera], ground_height, 1)
        groundPointsX = [gp.position[0] for gp in groundProjections]
        groundPointsY = [gp.position[1] for gp in groundProjections]
        groundPointsX.append(groundPointsX[0])
        groundPointsY.append(groundPointsY[0])

        ax.plot(groundPointsX, groundPointsY, 'k', linewidth=1)


fig, axs = plt.subplots(1, 1)
objectTrajectories = dict()
scenePointTrajectories = list()
plot_camera_groundview(simulationData['Cameras'], ground_height, axs)

for scenePoint in simulationData['Frame Data'][0]['Scene Points']:
    scenePointTrajectories.append({'x': [], 'y': []})

for i in range(0, len(handover_simulation_data), 3):

    for object in handover_simulation_data[i]['Valid Output Objects']:
        if object.id in objectTrajectories.keys():
            objectTrajectories[object.id]['x'].append(object.position[0])
            objectTrajectories[object.id]['y'].append(object.position[1])
        else:
            objectTrajectories[object.id] = dict()
            objectTrajectories[object.id]['x'] = [object.position[0]]
            objectTrajectories[object.id]['y'] = [object.position[1]]

    for iScenePoint, scenePoint in enumerate(simulationData['Frame Data'][i]['Scene Points']):
        scenePointTrajectories[iScenePoint]['x'].append(scenePoint.position[0])
        scenePointTrajectories[iScenePoint]['y'].append(scenePoint.position[1])

for scenePointTraj in scenePointTrajectories:
    axs.plot(scenePointTraj['x'], scenePointTraj['y'], 'k', alpha=0.15)

for objectId in objectTrajectories:
    axs.plot(objectTrajectories[objectId]['x'], objectTrajectories[objectId]['y'], '.-')

axs.axis('equal')
plt.show()


# ---------- Animations ----------
plot_3D_space = False
plot_camera_views = False
plot_projective_space = True
export_video = False


# Animation: 3D window
if plot_3D_space:
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
    if export_video:
        writervideo = animation.FFMpegWriter(fps=15)
        ani1.save('Simulation_3D_Space.mp4', writer=writervideo)


# Animation: Camera views (Under construction)
if plot_camera_views:
    # Define animation updater
    def animate_image_points(iteration, simulationData, scatters):
        for iCam in range(len(scatters)):
            imagePoints = [ip for ip in simulationData['Frame Data'][iteration]['Image Points'] if ip.camera_id == iCam]
            if len(imagePoints) > 0:
                scatters[iCam]._offsets = np.vstack([ip.position for ip in imagePoints])

    # Initialize plot. For each camera view, initialize scatter
    fig2, axs = plt.subplots(len(simulationData['Cameras']), 1)
    scatters = list()
    for iCam in range(len(simulationData['Cameras'])):
        # Set axis range
        axs[iCam].set_xlim((0, simulationData['Cameras'][iCam].resolution[0]))
        axs[iCam].set_ylim((0, simulationData['Cameras'][iCam].resolution[1]))
        #axs[iCam].invert_yaxis()
        #axs[iCam].title.set_text('Camera {}'.format(simulationData['Cameras'][iCam].id))

        # Initialize Scatters
        scatters.append(axs[iCam].scatter(500, 500))

    # Start animation
    ani2 = animation.FuncAnimation(fig2, animate_image_points, simulationData['Nr Frames'], fargs=(simulationData, scatters), interval=50, blit=False, repeat=True)
    if export_video:
        writervideo = animation.FFMpegWriter(fps=15)
        ani2.save('Simulation_Camera_Views.mp4', writer=writervideo)


# Animation: Handover points
if plot_projective_space:
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
    plot_camera_groundview(simulationData['Cameras'], ground_height, ax3[0])

    # Start animation
    ani3 = animation.FuncAnimation(fig3, animate_handover_points, simulationData['Nr Frames'], fargs=(simulationData, handover_simulation_data, scatters), interval=150, blit=False, repeat=True)
    if export_video:
        writervideo = animation.FFMpegWriter(fps=15)
        ani3.save('Simulation_Projective_Space.mp4', writer=writervideo)

plt.show()


