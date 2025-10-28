import numpy as np
import time
import pickle
import matplotlib as plt
from classes.Robot import Robot


def collect_data():
    """
    Collect data for velocity-based trajectory tracking.
    Moves the robot through a triangular path using velocity control.
    """
    
    # =============================================================================
    # SETUP AND INITIALIZATION
    # =============================================================================
    
    # Create robot object
    robot = Robot()
    
    # Define task-space waypoints [x, y, z, alpha] in mm and degrees
    ee_poses = np.array([
        [25, -100, 150, 0],    # Waypoint 1
        [150, 80, 300, 0],     # Waypoint 2
        [250, -115, 75, 0],    # Waypoint 3
        [25, -100, 150, 0]     # Return to Waypoint 1
    ])
    
    # TODO: Compute IK for all waypoints



    # Store results in joint_angles array (4x4)
    joint_angles = np.zeros((len(ee_poses), 4))
    for i in range(len(ee_poses)):
        try:
            joint_angles[i, :] = robot.get_ik(ee_poses[i, :])
        except ValueError as e:
            raise ValueError(f"End-Effector Pose Unreachable at Waypoint {i+1}: {e}")
    

    # =============================================================================
    # CONTROL PARAMETERS
    # =============================================================================
    
    velocity_des = 50.0      # Desired task-space speed (mm/s)
    tolerance = 5.0          # Convergence tolerance (mm)
    max_joint_vel = 45.0     # Maximum joint velocity limit (deg/s) for safety
    
    print(f"\nControl parameters:")
    print(f"  Desired velocity: {velocity_des} mm/s")
    print(f"  Tolerance: {tolerance} mm")
    
    # =============================================================================
    # DATA STORAGE PRE-ALLOCATION
    # =============================================================================
    
    # Pre-allocate arrays for data collection (over-allocate for safety)
    max_samples = 10000
    data_time = np.zeros(max_samples)
    data_q = np.zeros((max_samples, 4))              # Joint angles (deg)
    data_q_dot = np.zeros((max_samples, 4))          # Joint velocities (deg/s)
    data_ee_pos = np.zeros((max_samples, 5))         # End-effector pose [x,y,z,pitch,yaw]
    data_ee_vel_cmd = np.zeros((max_samples, 3))    # Commanded EE velocity (mm/s)
    data_ee_vel_actual = np.zeros((max_samples, 6)) # Actual EE velocity (mm/s, rad/s)
    count = 0  # Sample counter
    
    
    # =============================================================================
    # ROBOT INITIALIZATION
    # =============================================================================
    
    print("\nInitializing robot...")
    # Enable motors
    robot.write_motor_state(True)
    
    
    # TODO: Move to starting position using position control
    print("Moving to start position...")
    robot.write_mode("position")

    print("Moving to start position...")
    
    # Hint: write.time(), write_joints(), sleep
    # YOUR CODE HERE
    robot.write_time(5.0)
    robot.write_joints(joint_angles[0, :])
    time.sleep(5.0)

    
    
    # Switch to velocity control mode
    print("\nSwitching to velocity control mode...")
    robot.write_mode("velocity")
    time.sleep(0.5)
    
    
    # =============================================================================
    # VELOCITY-BASED TRAJECTORY TRACKING
    # =============================================================================
    
    print("\nStarting velocity-based trajectory tracking...")
    start_time = time.perf_counter()
    
    # Loop through waypoints 2, 3, 4 (indices 1, 2, 3)
    for i in range(1, len(ee_poses)):
        
        # Extract target position (first 3 elements: x, y, z)
        p_target = ee_poses[i][:3]
        print(f"\n--- Moving to Waypoint {i+1}: {p_target} ---")
        
        # Initialize distance to target
        distance = np.inf
        iteration = 0
        
        # Continue until within tolerance of target
        while distance > tolerance:
            
            loop_start = time.perf_counter()
            
            # -----------------------------------------------------------------
            # STEP 1: READ CURRENT STATE
            # -----------------------------------------------------------------
            
            # Read current joint angles and velocities
            q_deg = robot.get_joints_readings()[0]
            q_dot_deg = robot.get_joints_readings()[1]

            # Convert joint velocities from deg/s to rad/s
            q_dot_rad = np.deg2rad(q_dot_deg)
            
            # Get current end-effector pose
            ee_pose = robot.get_ee_pos()
            p_current = ee_pose[:3]
            
            
            # -----------------------------------------------------------------
            # STEP 2: COMPUTE DISTANCE AND DIRECTION TO TARGET
            # -----------------------------------------------------------------
            
            # Compute error vector (target - current position)
            error = (p_target - p_current)  # Replace with calculation
            
            #Compute distance to target (norm of error vector)
            distance = np.linalg.norm(error)  # Replace with calculation

            # Compute unit direction vector: direction = error / distance (avoid division by zero)
            direction = error/distance  # Replace with calculation
            
            # -----------------------------------------------------------------
            # STEP 3: GENERATE DESIRED VELOCITY
            # -----------------------------------------------------------------
            
            # Scale direction by desired speed
            v_des = velocity_des*direction  # Replace with calculation
            
            # Form 6D desired velocity vector [v_x, v_y, v_z, omega_x, omega_y, omega_z]
            # Hint: Stack v_des with zeros for angular velocity
            #Form a 6 × 1 vector  ̇pdes = [vdes; 03×1] (zero angular velocity)
            p_dot_des = np.hstack((v_des, np.zeros(3)))  # Replace with calculation


            # -----------------------------------------------------------------
            # STEP 4: INVERSE VELOCITY KINEMATICS
            # -----------------------------------------------------------------
            
            # Get Jacobian at current configuration
            J = robot.get_jacobian(q_deg)
            
            # Compute pseudo-inverse of Jacobian
            J_pinv = np.linalg.pinv(J)
            
            # Compute required joint velocities (rad/s)
    
            q_dot_cmd_rad = J_pinv @ p_dot_des  # Replace with calculation
            
            # Convert joint velocities from rad/s to deg/s
            q_dot_cmd_deg = np.rad2deg(q_dot_cmd_rad)  # Replace with conversion
            
            
            # -----------------------------------------------------------------
            # STEP 5: SEND VELOCITY COMMAND TO ROBOT
            # -----------------------------------------------------------------
            
            #Send velocity command to robot
            # Hint: Use robot.write_velocities(q_dot_cmd_deg)
            robot.write_velocities(q_dot_cmd_deg)

            

            # -----------------------------------------------------------------
            # STEP 6: VERIFY WITH FORWARD VELOCITY KINEMATICS
            # -----------------------------------------------------------------
            
            #Compute actual end-effector velocity
            # Hint: Use robot.get_fwd_vel_kin(...)
            p_dot_actual = robot.get_fwd_vel_kin(q_deg, q_dot_deg)
            
            
            # -----------------------------------------------------------------
            # STEP 7: DATA COLLECTION
            # -----------------------------------------------------------------
            
            if count < max_samples:
                data_time[count] = time.perf_counter() - start_time
                data_q[count, :] = q_deg
                data_q_dot[count, :] = q_dot_deg
                data_ee_pos[count, :] = ee_pose
                data_ee_vel_cmd[count, :] = v_des
                data_ee_vel_actual[count, :] = p_dot_actual
                count += 1
            
            iteration += 1
            
        
        # End of while loop - target reached
        print(f"  Reached Waypoint {i+1}! Final distance: {distance:.2f} mm")
        
        # Stop robot briefly at waypoint
        # Hint: Send zero velocities and sleep briefly
        robot.write_velocities(np.zeros(4))
        time.sleep(1.0)
        
    
    
    # =============================================================================
    # CLEANUP AND DATA SAVING
    # =============================================================================
    
    #Stop robot completely
    print("\nTrajectory complete! Stopping robot...")
    robot.write_velocities(np.zeros(4))
    
    total_time = time.perf_counter() - start_time
    print(f"\nTotal execution time: {total_time:.2f} s")
    print(f"Total samples collected: {count}")
    print(f"Average sample rate: {count/total_time:.1f} Hz")
    
    # Trim unused portions of pre-allocated arrays
    data_time = data_time[:count]
    data_q = data_q[:count, :]
    data_q_dot = data_q_dot[:count, :]
    data_ee_pos = data_ee_pos[:count, :]
    data_ee_vel_cmd = data_ee_vel_cmd[:count, :]
    data_ee_vel_actual = data_ee_vel_actual[:count, :]
    
    # Save all data to pickle file
    filename='lab5_4_data.pkl'
    # Create dictionary with all collected data and control parameters
    print(f"\nSaving data to {filename}...")
    # YOUR CODE HERE
    data_dict = {
        'time': data_time,
        'joint_angles': data_q,
        'joint_velocities': data_q_dot,
        'ee_positions': data_ee_pos,
        'ee_vel_cmd': data_ee_vel_cmd,
        'ee_vel_actual': data_ee_vel_actual,
        'waypoints': ee_poses,
        'velocity_des': velocity_des,
        'tolerance': tolerance,
        'max_joint_vel': max_joint_vel
    }
    
    # Write dictionary to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print("Data saved successfully!")

def plot_data(dat):
    """
    Load data and create required plots.
    """
    

    """Load and plot data from pickle file"""
    # Load data
    with open(filename, "rb") as f:
        data = pickle.load(f)
        
    
    timestamps_np = data["timestamps_s"]
    joint_angles_np = data["joint_deg"]
    ee_positions_np = data["ee_pos_mm"]
    waypoints_np = data["waypoints_deg"]
    
    # Plot joint angles
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)
    
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(timestamps_np, joint_angles_np[:, i], label=f"Joint {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(f"Joint {i+1} Angle vs Time")
        plt.grid()
        plt.legend()
    plt.show()

    # Plot X and Z position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps_np, ee_positions_np[:, 0], 'b-', label='x position')
    plt.plot(timestamps_np, ee_positions_np[:, 2], 'r-', label='z position')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (mm)')
    plt.title('End-Effector Position vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot X-Z trajectory with waypoints
    plt.figure(figsize=(8, 8))
    plt.plot(ee_positions_np[:, 0], ee_positions_np[:, 2], 'b-', label='Actual Path')
    plt.plot(ee_positions_np[0, 0], ee_positions_np[0, 2], 'go', label='Start')
    
    # Create robot instance to compute waypoint positions
    robot = Robot()
    waypoint_ee_pos = []
    for waypoint in waypoints_np:
        T = robot.get_fk(waypoint)
        waypoint_ee_pos.append([T[0,3], T[2,3]])  # x,z coordinates
    waypoint_ee_pos = np.array(waypoint_ee_pos)
    
    plt.plot(waypoint_ee_pos[:, 0], waypoint_ee_pos[:, 1], 'ro', label='Waypoints')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Z Position (mm)')
    plt.title('End-Effector X-Z Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Plot X-Y trajectory with waypoints
    plt.figure(figsize=(8, 8))
    plt.plot(ee_positions_np[:, 0], ee_positions_np[:, 1], 'b-', label='Actual Path')
    plt.plot(ee_positions_np[0, 0], ee_positions_np[0, 1], 'go', label='Start')
    plt.plot(waypoint_ee_pos[:, 0], [T[1,3] for T in [robot.get_fk(w) for w in waypoints_np]], 'ro', label='Waypoints')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('End-Effector X-Y Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()



    # plot 3D plot showing path traced by end-effector in task space during entire trajectory, marking the 3 waypoints and their coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data_ee_pos[:, 0], data_ee_poses[:, 1], data_ee_poses[:, 2], 'b-', label='End-Effector Path')
    ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], waypoints_np[:, 2], c='r', marker='o', s=100, label='Waypoints')
    for i, waypoint in enumerate(waypoints_np):
        ax.text(waypoint[0], waypoint[1], waypoint[2], 
            f'Waypoint {i}\n({waypoint[0]:.0f}, {waypoint[1]:.0f}, {waypoint[2]:.0f})', 
            color='black')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_zlabel('Z Position (mm)')
        ax.set_title('End-Effector 3D Trajectory')
        ax.legend()
        plt.show()
    


    #Joint angles from sensor readings. Plot four lines showing measured joint angles q1, q2, q3, q4 (degrees) versus time (sec). These are the angles recorded via get_joints_readings() during motion. Use distinguishable line styles and include a legend
    # Create joint angle plots
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(timestamps_np, data_q[:, i], label=f"Joint {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(f"Joint {i+1} Angle vs Time")
        plt.grid()
        plt.legend()
        plt.show()


#Task-space pose, velocity, and acceleration. Create a figure with three 2D subplots stacked vertically:– Top: Four lines showing x, y, z (mm) and ¸ (deg) versus time (sec)– Middle: Four lines showing ˙x, ˙y, ˙z (mm/s) and ˙¸ (deg/s) versus time– Bottom: Four lines showing ¨x, ¨y, ¨z (mm/s2) and ¨¸ (deg/s2) versus time. Use distinguishable line styles and include legends. Hint: Compute velocities and accelerations using numerical differentiation (e.g., np.gradient() or finite differences).
    # Task-space pose, velocity, and acceleration plotting

    # Compute velocities and accelerations
    ee_velocities = np.gradient(data_ee_poses, axis=0)  # mm/s
    ee_accelerations = np.gradient(ee_velocities, axis=0)  # mm/s^2

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)

    # Top subplot: Position
    plt.subplot(3, 1, 1)
    plt.plot(timestamps_np, data_ee_poses[:, 0], label='x (mm)')
    plt.plot(timestamps_np, data_ee_poses[:, 1], label='y (mm)')
    plt.plot(timestamps_np, data_ee_poses[:, 2], label='z (mm)')
    plt.plot(timestamps_np, data_ee_poses[:, 3], label='¸ (deg)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End-Effector Position vs Time')
    plt.grid(True)
    plt.legend()

    # Middle subplot: Velocity
    plt.subplot(3, 1, 2)
    plt.plot(timestamps_np, ee_velocities[:, 0], label='˙x (mm/s)')
    plt.plot(timestamps_np, ee_velocities[:, 1], label='˙y (mm/s)')
    plt.plot(timestamps_np, ee_velocities[:, 2], label='˙z (mm/s)')
    plt.plot(timestamps_np, ee_velocities[:, 3], label='˙¸ (deg/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('End-Effector Velocity vs Time')
    plt.grid(True)
    plt.legend()

    # Bottom subplot: Acceleration
    plt.subplot(3, 1, 3)
    plt.plot(timestamps_np, ee_accelerations[:, 0], label='¨x (mm/s²)')
    plt.plot(timestamps_np, ee_accelerations[:, 1], label='¨y (mm/s²)')
    plt.plot(timestamps_np, ee_accelerations[:, 2], label='¨z (mm/s²)')
    plt.plot(timestamps_np, ee_accelerations[:, 3], label='¨¸ (deg/s²)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title('End-Effector Acceleration vs Time')
    plt.grid(True)
    plt.legend()

    plt.show()




if __name__ == "__main__":
    # Run data collection
    collect_data()
    # Plot data
    plot_data()