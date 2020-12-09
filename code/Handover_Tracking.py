

def output_object_tracking(output_objects, clusters, deltaTime):

    # Define model
    F = np.array([[1, 0, deltaTime, 0], [0, 1, 0, deltaTime], [0, 0, 1, 0], [0, 0, 0, 1]])  # Process model
    Q = np.diag([0, 0, 1, 1]) * 1  # Process noise

    # Kalman prediction of output_object states
    predictedStates = list()
    predictedStateCov = list()
    for object in output_objects:
        predictedStates.append(np.dot(F, object.get_state()))

