def get_jacobian(self, q):
    """
    Compute the manipulator Jacobian matrix.
    
    Parameters:
    -----------
    q : numpy.ndarray
        Joint angles in DEGREES [q1, q2, q3, q4]
    
    Returns:
    --------
    J : numpy.ndarray (6, 4)
        Manipulator Jacobian matrix where:
        - J[0:3, :] = J_v (linear velocity Jacobian) - units: mm
        - J[3:6, :] = J_ω (angular velocity Jacobian) - dimensionless
        
        Each column i corresponds to joint i's contribution:
        J[:, i] = [z_{i-1} × (o_4 - o_{i-1}); z_{i-1}]
    
    Notes:
    ------
    - Accepts angles in degrees for consistency with other methods
    - Internally converts to radians for computation
    - The Jacobian derivation assumes radians due to differential calculus

    """
    
    # Step 1: Compute accumulated transformation matrices from each joint to the base frame.
    a_matrices = self.get_int_mat(q) # note: q's get converted internally to radians for computation
    t_matrices = np.zeros((4, 4, 4))
    t_matrices[:, :, 0] = a_matrices[:, :, 0]  # T_1^0 = A_1

    for i in range(1, 4):
        t_matrices[:, :, i] = t_matrices[:, :, i-1] @ a_matrices[:, :, i]

    # Initialize Jacobian
    J = np.zeros((6, 4))
    
    # Extract end-effector origin (o_4)
    o_ee = t_matrices[:3, 3, 3]  # Last T matrix, position column
    
    # Construct origins array: [o_0, o_1, o_2, o_3]
    # o_0 is at base (origin), o_i extracted from T_i^0
    origins = np.hstack((
        np.zeros((3, 1)),           # o_0 = [0, 0, 0]^T
        t_matrices[:3, 3, :3]       # [o_1, o_2, o_3]
    ))
    
    # Construct z-axes array: [z_0, z_1, z_2, z_3]
    # z_0 is base frame z-axis, z_i extracted from T_i^0 (3rd column)
    z_axes = np.hstack((
        np.array([[0, 0, 1]]).T,    # z_0 = [0, 0, 1]^T
        t_matrices[:3, 2, :3]       # [z_1, z_2, z_3]
    ))
    
    # Compute each column of the Jacobian
    for i in range(4):
        # Linear velocity component: z_i × (o_ee - o_i)
        J_v_i = np.cross(z_axes[:, i], o_ee - origins[:, i])
        
        # Angular velocity component: z_i
        J_omega_i = z_axes[:, i]
        
        # Stack to form column i
        J[:, i] = np.hstack((J_v_i, J_omega_i))
    
    return J