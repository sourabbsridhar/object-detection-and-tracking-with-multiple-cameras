import numpy as np


# Create cameras
K = np.matrix('50, 0, 960; 0, 20, 540; 0, 0, 1')

R_norm = np.matrix('0, 1, 0; 1, 0, 0; 0, 0, -1')
t_norm = np.matrix('0; 0; 2')

P_norm = np.concatenate((R_norm, t_norm), axis=1)
C = -R_norm.T * t_norm
P = K*P_norm

print(P)

point = np.matrix('1; 1; 0; 1')
proj_point = P*point
proj_point = proj_point / proj_point[2]
print(proj_point)
