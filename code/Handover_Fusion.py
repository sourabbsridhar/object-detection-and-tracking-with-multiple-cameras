import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy


class Projection:
    def __init__(self, position, velocity, detection_index, camera_index):
        self.position = position
        self.velocity = velocity
        self.detection_index = detection_index
        self.camera_index = camera_index

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

    def __eq__(self, other):
        if type(other) != Cluster:
            return False
        elif self.index == other.index:
            return False
        return True

    def add_projection(self, projection):
        self.projections.append(projection)

    def validate_cluster(self, R, V):

        for projection in self.projections:
            # Make sure projection isn't farther than R/2 and V/2 away from the average and that no
            if np.linalg.norm(projection.get_position() - self.get_position()) > R/2 or \
                    np.linalg.norm(projection.get_velocity() - self.get_velocity()) > V/2:
                return False

            # Make sure no there are no projections from the same camera
            for another_projection in self.projections:
                if another_projection != projection and another_projection.camera_index == projection.camera_index:
                    return False

        return True

    def get_position(self):
        return np.mean([projection.position for projection in self.projections]) if len(projections) > 0 else None

    def get_velocity(self):
        return np.mean([projection.velocity for projection in self.projections]) if len(projections) > 0 else None


def objective_function(projection1, projection2, R, V):
    position_distance = np.linalg.norm(projection1.get_position() - projection2.get_position())
    velocity_distance = np.linalg.norm(projection1.get_velocity() - projection2.get_velocity())

    return (position_distance / R)**2 + (velocity_distance / V)**2


def fusion(projections, R, V):

    clusters = []
    clustered_projections = []

    change = True
    while change:
        change = False

        sigma_list = [elem for elem in clusters + projections if elem not in clustered_projections]

        for theta in sigma_list:
            for sigma in sigma_list:

                # Make sure theta and sigma are not the same element
                if theta == sigma:
                    continue

                # Condition 1: Make sure that theta and sigma aren't projections from the same camera
                if type(theta) == Projection and type(sigma) == Projection:
                    if theta.camera_index == sigma.camera_index:
                        continue

                # Condition 2: Make sure that theta and sigma are within R and V from each other
                if np.linalg.norm(theta.get_position() - sigma.get_position()) > R \
                        or np.linalg.norm(theta.get_velocity() - sigma.get_velocity()) > V:
                    continue


                # Condition 3: Make sure that theta is the minimizer of the objective function for sigma
                argmin_sigma = np.argmin([objective_function(sigma, gamma, R, V)
                                          if sigma != gamma else np.inf for gamma in sigma_list])
                if theta != sigma_list[argmin_sigma]:
                    continue

                # Condition 4: Make sure that sigma is the minimizer of the objective function for theta
                argmin_theta = np.argmin([objective_function(theta, gamma, R, V)
                                          if theta != gamma else np.inf for gamma in sigma_list])
                if sigma != sigma_list[argmin_theta]:
                    continue

                # All conditions are met

                # If both theta and sigma are projections
                if type(theta) == Projection and type(sigma) == Projection:
                    # Add new cluster
                    newCluster = Cluster(len(clusters))
                    newCluster.add_projection(theta)
                    newCluster.add_projection(sigma)
                    clusters.append(newCluster)

                    # Add theta and sigma to clustered projection set so that they are not evaluated anymore
                    clustered_projections.append(theta)
                    clustered_projections.append(sigma)
                    change = True
                    break

                # If one of theta or sigma is a projection and the other a cluster
                elif type(theta) == Projection or type(sigma) == Projection:
                    aCluster = theta if type(theta) == Cluster else sigma
                    aProjection = theta if type(theta) == Projection else sigma

                    testCluster = deepcopy(aCluster)
                    testCluster.add_projection(aProjection)

                    if testCluster.validate_cluster(R,V) == True:
                        aCluster.add_projection(aProjection)
                        clustered_projections.append(aProjection)
                        change = True
                        break

                # If both theta and sigma are clusters
                else:
                    testCluster = Cluster(len(clusters))
                    testCluster.projections = theta.projections + sigma.projections
                    if testCluster.validate_cluster(R, V) == True:
                        # Remove theta and sigma from clusters
                        clusters = [cluster for cluster in clusters if cluster != theta and cluster != sigma]
                        # Add new cluster
                        clusters.append(testCluster)
                        # Re-index clusters
                        for i,cluster in enumerate(clusters): cluster.index = i+1

                        change = True
                        break

            if change == True:
                break

    # Add all projections that did not get clustered into individual clusters
    for projection in projections:
        if projection not in clustered_projections:
            newCluster = Cluster(len(clusters))
            newCluster.add_projection(projection)
            clusters.append(newCluster)

    return clusters

nr_points = 40
nr_cameras = 6
R = 1
V = 1

# Generate random projections from various cameras
projectionCameraIndexes = [random.randrange(nr_cameras) for i in range(nr_points)]
projectionDetectionIndexes = [projectionCameraIndexes[0:i].count(projectionCameraIndexes[i]) for i in range(nr_points)]


projections = [Projection(np.random.uniform(0, 1, 2), np.random.uniform(0, 1, 2),
                          projectionDetectionIndexes[i], projectionCameraIndexes[i]) for i in range(nr_points)]
projections.sort(key=lambda projection: (projection.camera_index, projection.detection_index))
for projection in projections:
    print(projection)


# Perform fusion
clusters = fusion(projections, R, V)

for cluster in clusters:
    if len(cluster.projections) > 2:
        print('Cluster {} has {} projections'.format(cluster.index, len(cluster.projections)))

# Plot
fig, axs = plt.subplots(1, 2)
legends = []
markers = ['o', 'x', 'P', '^', 's', '*']
for counter, projection in enumerate(projections):

    marker = markers[projection.camera_index]
    axs[0].plot(projection.position[0], projection.position[1], marker)
    axs[1].plot(projection.velocity[0], projection.velocity[1], marker)

    legends.append('id {}, cam {}'.format(projection.detection_index, projection.camera_index))
axs[0].title.set_text('Positions')
axs[1].title.set_text('Velocities')
axs[0].legend(legends, title='Projections', bbox_to_anchor=(-0.05, 1.1), loc='upper right')
axs[0].axis('equal')
axs[1].axis('equal')


fig, axs = plt.subplots(1, 2)
legends = []
for counter, cluster in enumerate(clusters):
    if np.floor(counter/10) % 5 == 0:
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], 'o')
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], 'o')
    elif np.floor(counter/10) % 5 == 1:
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], 'x')
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], 'x')
    elif np.floor(counter/10) % 5 == 2:
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], 'P')
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], 'P')
    elif np.floor(counter/10) % 5 == 3:
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], '^')
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], '^')
    else:
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], 's')
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], 's')
    legends.append('{}'.format(cluster.index))
axs[0].title.set_text('Positions')
axs[1].title.set_text('Velocities')
axs[0].legend(legends, title='Clusters', bbox_to_anchor=(-0.05, 1), loc='upper right')
axs[0].axis('equal')
axs[1].axis('equal')

plt.show()
print('Placeholder')