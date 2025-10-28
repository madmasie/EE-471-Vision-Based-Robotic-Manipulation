# alidate your Jacobian implementation using two test configurations provided in Table 1:
# 1. Overhead singularity configuration: q = [0; −10:62; −79:38; 0] degrees. This configuration
# places the arm fully extended along the z0 axis (directly overhead), creating a kinematic singu-
# larity.
# 2. Home configuration: q = [0; 0; 0; 0] degrees. This is the arm’s zero position.
# For each configuration, your script should:
# • Call get_jacobian(q) to compute the full Jacobian matrix
# • Extract the linear velocity Jacobian Jv (upper-left 3 × 3 submatrix)
# • Compute the determinant of Jv using np.linalg.det()
# • Display both the complete 6 × 4 Jacobian matrix and the determinant value
# • Format output clearly with appropriate labels and precision (3-4 decimal places)

#test case q (deg):         Jacobian J(q)
#[0, -10.62, -79.38, 0]     [[0, 380, 250, 126],
#                            [0.042, 0, 0, 0],
#                            [0, -0.042, 0, 0],
#                            [0, 0, 0, 0],
#                            [0, 1, 1, 1],
#                            [1, 0, 0, 0]]

# [0,0,0,0]                 #compute 

import numpy as np
from classes.Robot import Robot


#printing less decimals for clarity
np.set_printoptions(precision=4, suppress=True)

#call robot class
bot = Robot()

#the test joint configurations (in degrees)
test_configs = {
    "Overhead Singularity": np.array([0.0, -10.62, -79.38, 0.0]),
    "Home Configuration":   np.array([0.0, 0.0, 0.0, 0.0])
}

#loops through each q_deg test configuration
for name, q_deg in test_configs.items():
    #compute Jacobian (6x4)
    J = bot.get_jacobian(q_deg)

    #get linear velocity Jacobian (upper-left 3x3 block)
    Jv = J[0:3, 0:3]

    #determinant of Jv
    determinant = np.linalg.det(Jv)

    #printing results
    print(f"\n{name} (q = {q_deg.tolist()} degrees)")
    print("=" * 60)
    print("Full Jacobian J(q) [6x4]:")
    print(J)
    print(f"\nDeterminant of Jv (upper-left 3x3): {determinant:.4f}")
    print("=" * 60)