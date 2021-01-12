import numpy as np
from copy import deepcopy
from Handover_Library import Cluster, Projection

def objective_function(projection1, projection2, R, V):
    position_distance = np.linalg.norm(projection1.get_position() - projection2.get_position())
    velocity_distance = np.linalg.norm(projection1.get_velocity() - projection2.get_velocity())

    return (position_distance / R)**2 + (velocity_distance / V)**2


def fusion(projections, R, V):

    clusters = []
    clustered_projections = []

    changes_made = True
    strict_conditions = True
    while changes_made:
        changes_made = False

        sigma_list = [elem for elem in clusters + projections if elem not in clustered_projections]

        for i_theta, theta in enumerate(sigma_list):

            # Pre-computation for condition 4 (only for computational efficiency)
            if strict_conditions:
                argmin_theta = np.argmin([objective_function(theta, gamma, R, V)
                                          if theta != gamma else np.inf for gamma in sigma_list])

            for sigma in sigma_list[i_theta+1:]:

                # Condition 1-2: Make sure that theta and sigma aren't or don't contain projections from the same camera
                if type(theta) == Projection and type(sigma) == Projection:
                    if theta.camera_index == sigma.camera_index:
                        continue
                elif type(theta) == Projection or type(sigma) == Projection:
                    a_cluster = theta if type(theta) == Cluster else sigma
                    a_projection = theta if type(theta) == Projection else sigma
                    test_cluster = deepcopy(a_cluster)
                    test_cluster.add_projection(a_projection)
                    if not test_cluster.validate_cluster(R, V):
                        continue
                else:
                    test_cluster = Cluster(len(clusters))
                    test_cluster.projections = theta.projections + sigma.projections
                    test_cluster.update_averages()
                    if not test_cluster.validate_cluster(R, V):
                        continue

                # Condition 3: Make sure that theta and sigma are within R and V from each other
                if np.linalg.norm(theta.get_position() - sigma.get_position()) > R \
                        or np.linalg.norm(theta.get_velocity() - sigma.get_velocity()) > V:
                    continue

                # Condition 3: Make sure that theta is the minimizer of the objective function for sigma
                if strict_conditions:
                    argmin_sigma = np.argmin([objective_function(sigma, gamma, R, V)
                                              if sigma != gamma else np.inf for gamma in sigma_list])
                    if theta != sigma_list[argmin_sigma]:
                        continue

                # Condition 4: Make sure that sigma is the minimizer of the objective function for theta
                if strict_conditions:
                    # argmin_theta was computed outside of sigma-loop
                    if sigma != sigma_list[argmin_theta]:
                        continue

                # All conditions are met

                # If both theta and sigma are projections
                if type(theta) == Projection and type(sigma) == Projection:
                    # Add new cluster
                    new_cluster = Cluster(len(clusters))
                    new_cluster.add_projection(theta)
                    new_cluster.add_projection(sigma)
                    clusters.append(new_cluster)

                    # Add theta and sigma to clustered projection set so that they are not evaluated anymore
                    clustered_projections.append(theta)
                    clustered_projections.append(sigma)
                    changes_made = True
                    break

                # If one of theta or sigma is a projection and the other a cluster
                elif type(theta) == Projection or type(sigma) == Projection:
                    if type(theta) == Cluster:
                        theta.add_projection(sigma)
                        clustered_projections.append(sigma)
                    else:
                        sigma.add_projection(theta)
                        clustered_projections.append(theta)
                    changes_made = True
                    break

                # If both theta and sigma are clusters
                else:
                    new_cluster = Cluster(len(clusters))
                    new_cluster.projections = theta.projections + sigma.projections
                    new_cluster.update_averages()

                    # Remove theta and sigma from clusters
                    clusters = [cluster for cluster in clusters if cluster != theta and cluster != sigma]
                    # Add new cluster
                    clusters.append(new_cluster)
                    # Re-index clusters
                    for i, cluster in enumerate(clusters): cluster.index = i

                    changes_made = True
                    break

            if changes_made:
                break

        if changes_made:
            strict_conditions = True
        elif strict_conditions:
            changes_made = True
            strict_conditions = False

    # Add all projections that did not get clustered into individual clusters
    for projection in projections:
        if projection not in clustered_projections:
            newCluster = Cluster(len(clusters))
            newCluster.add_projection(projection)
            clusters.append(newCluster)

    return clusters




if __name__ == '__main__':

    # Imports for testing and plotting, not necessary for clustering
    import random
    import matplotlib.pyplot as plt

    # Testing
    nr_points = 20
    nr_cameras = 3
    R = 1
    V = 1

    # Generate random projections from various cameras
    projectionCameraIndexes = [random.randrange(nr_cameras) for i in range(nr_points)]
    projectionDetectionIndexes = [projectionCameraIndexes[0:i].count(projectionCameraIndexes[i]) for i in range(nr_points)]

    projections = [Projection(np.random.uniform(-1, 1, 2), np.random.uniform(-1, 1, 2),
                              projectionDetectionIndexes[i], projectionCameraIndexes[i]) for i in range(nr_points)]
    projections.sort(key=lambda projection: (projection.camera_index, projection.detection_index))

    # Perform fusion
    clusters = fusion(projections, R, V)

    # Perform Kalman Tracking
    output_objects = list()
    output_objects = output_object_tracking(output_objects, clusters, 0.1)

    # Print if any clusters are larger than 2
    for cluster in clusters:
        if len(cluster.projections) > 2:
            print('Cluster {} has {} projections'.format(cluster.index, len(cluster.projections)))

    # Plot Projections
    legends = []
    markers = ['o', '1', 's', 'p', 'v', 'P', '*', 'h', 'X', 'D', '<', '>']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    size = 10

    fig, axs = plt.subplots(1, 2)
    for counter, projection in enumerate(projections):
        mycolor = colors[projection.camera_index]
        mymarker = markers[min(projection.detection_index,len(markers)-1)]
        axs[0].plot(projection.position[0], projection.position[1], mymarker, color=mycolor, markersize=size)
        axs[1].plot(projection.velocity[0], projection.velocity[1], mymarker, color=mycolor, markersize=size)
        legends.append('id {}, cam {}'.format(projection.detection_index, projection.camera_index))

    axs[0].title.set_text('Positions')
    axs[1].title.set_text('Velocities')
    axs[0].legend(legends, title='Projections', bbox_to_anchor=(-0.05, 1.1), loc='upper right')
    axs[0].axis('equal')
    axs[1].axis('equal')


    # Plot Cluster projections and clusters
    fig, axs = plt.subplots(1, 2)
    fig2, axs2 = plt.subplots(1, 2)

    legends = []
    for counter, cluster in enumerate(clusters):
        marker_index = int(np.floor(counter/10))
        axs[0].plot([proj.position[0] for proj in cluster.projections], [proj.position[1] for proj in cluster.projections], markers[marker_index], markersize=size)
        axs[1].plot([proj.velocity[0] for proj in cluster.projections], [proj.velocity[1] for proj in cluster.projections], markers[marker_index], markersize=size)

        cluster_pos = cluster.get_position()
        cluster_vel = cluster.get_velocity()
        axs2[0].plot(cluster_pos[0], cluster_pos[1], markers[marker_index], markersize=size)
        axs2[1].plot(cluster_vel[0], cluster_vel[1], markers[marker_index], markersize=size)
        legends.append('{}'.format(cluster.index))

    axs[0].title.set_text('Positions')
    axs[1].title.set_text('Velocities')
    axs[0].legend(legends, title='Clusters', bbox_to_anchor=(-0.05, 1), loc='upper right')
    axs[0].axis('equal')
    axs[1].axis('equal')

    axs2[0].title.set_text('Positions')
    axs2[1].title.set_text('Velocities')
    axs2[0].legend(legends, title='Clusters', bbox_to_anchor=(-0.05, 1), loc='upper right')
    axs2[0].axis('equal')
    axs2[1].axis('equal')


    plt.show()