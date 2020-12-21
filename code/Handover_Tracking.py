import numpy as np
from Handover_Library import OutputObject

def output_object_tracking(output_objects, clusters, output_objectCount, deltaTime, distance_maximum, observation_loss_maximum, observation_new_minimum):

    next_output_objects = list()

    # Define model
    F = np.array([[1, 0, deltaTime, 0], [0, 1, 0, deltaTime], [0, 0, 1, 0], [0, 0, 0, 1]])  # Process model
    Q = np.diag([0, 0, 1, 1]) * 1  # Process noise covariance
    H = np.identity(4)  # Observation model
    R = np.diag([20, 20, 40, 40])  # Observation noise covariance

    # Kalman prediction of output_object states
    predicted_states = list()
    predicted_state_cov = list()
    for object in output_objects:
        predicted_states.append(F.dot(object.get_state()))
        predicted_state_cov.append(F.dot(object.estimate_covariance).dot(F.T) + Q)

    # Calculate squared Mahalanobis distance from every cluster to every object prediction
    distances = np.zeros((len(clusters), len(output_objects)))
    for iObject, object in enumerate(output_objects):
        for iCluster, cluster in enumerate(clusters):
            objState = object.get_state()
            clusterState = cluster.get_state()
            error = (objState - clusterState)
            #distance = (error.T).dot(np.linalg.inv(predicted_state_cov[iObject])).dot(error)
            distance = (error.T).dot(np.linalg.inv(R)).dot(error)
            distances[iCluster, iObject] = distance

    # Match clusters to objects if Mahalanobis distance is mutual minimum and under a maximum distance
    matches = list()
    change = True
    while change and not np.all(np.isnan(distances)):
        change = False
        for iObject in range(len(output_objects)):

            if not np.all(np.isnan(distances[:, iObject])):
                bestClusterArg = np.nanargmin(distances[:, iObject])
            else:
                continue

            if distances[bestClusterArg, iObject] < distance_maximum and iObject == np.nanargmin(distances[bestClusterArg, :]):
                matches.append([bestClusterArg, iObject])
                distances[:, iObject] = None
                distances[bestClusterArg, :] = None
                change = True
                if np.all(np.isnan(distances)):
                    break

    # For every object matched with a cluster, perform Kalman update using cluster as an observation and add to next
    # output object list. Also, if object is not validated, increase new observation count by 1 and check if valid
    for match in matches:
        iCluster, iObject = match

        # Calculate innovation and innovation covariance
        y = clusters[iCluster].get_state() - H.dot(predicted_states[iObject])
        S = H.dot(predicted_state_cov[iObject]).dot(H.T) + R

        # Calculate optimal Kalman gain
        K = predicted_state_cov[iObject].dot(H.T).dot(np.linalg.inv(S))

        # Calculate updated state estimate and state estimate covariance and save to the object
        output_objects[iObject].set_state(predicted_states[iObject] + K.dot(y))
        output_objects[iObject].estimate_covariance = (np.identity(4) - K.dot(H)).dot(predicted_state_cov[iObject])

        # Check validation count of object
        if output_objects[iObject].isValid is False:
            output_objects[iObject].observation_new_count += 1
            if output_objects[iObject].observation_new_count >= observation_new_minimum:
                output_objects[iObject].isValid = True

        # Reset objects observation loss count since it has been observed
        output_objects[iObject].observation_loss_count = 0

        # Add to next output objects
        next_output_objects.append(output_objects[iObject])

    # For every object without a matched cluster, increase observation loss count and add object to next output objects
    # if count is under maximum observation loss count, with its state and cov as predicted state cov.
    for iObject in [i for i in range(len(output_objects)) if i not in [match[1] for match in matches]]:
        output_objects[iObject].observation_loss_count += 1
        if output_objects[iObject].observation_loss_count < observation_loss_maximum:
            output_objects[iObject].set_state(predicted_states[iObject])
            output_objects[iObject].estimate_covariance = predicted_state_cov[iObject]
            next_output_objects.append(output_objects[iObject])

    # For every cluster that has not been matched, create a new object
    for iCluster in [i for i in range(len(clusters)) if i not in [match[0] for match in matches]]:
        next_output_objects.append(OutputObject('{}'.format(output_objectCount), clusters[iCluster].get_position(), clusters[iCluster].get_velocity()))
        output_objectCount += 1

    return (next_output_objects, output_objectCount)