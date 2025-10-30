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
        'ee_velocities_commanded': data_ee_vel_cmd,
        'ee_velocities_actual': data_ee_vel_actual,
        'waypoints': ee_poses,
        'velocity_des': velocity_des,
        'tolerance': tolerance,
        'max_joint_vel': max_joint_vel



    }
    
    # Write dictionary to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print("Data saved successfully!")
def plot_data(filename='lab5_4_data.pkl'):
    """
    Load data and create required plots per lab requirements.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Load data
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Extract data using correct keys
    timestamps_np = data["time"]
    joint_angles_np = data["joint_angles"] 
    joint_velocities_np = data["joint_velocities"]  # This is what we need for plot (b)
    ee_positions_np = data["ee_positions"]
    ee_velocities_cmd_np = data["ee_velocities_commanded"]
    ee_velocities_actual_np = data["ee_velocities_actual"]  # This is from get_fwd_vel_kin()
    waypoints_np = data["waypoints"]
    
    # a) 3D end-effector trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions_np[:, 0], ee_positions_np[:, 1], ee_positions_np[:, 2], 'b-', 
            linewidth=2, label='Actual End-Effector Path')
    ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], waypoints_np[:, 2], 
               c='red', marker='o', s=100, label='Waypoints')
    
    # Add waypoint labels
    for i, waypoint in enumerate(waypoints_np):
        ax.text(waypoint[0], waypoint[1], waypoint[2], 
                f'WP{i+1}\n({waypoint[0]:.0f},{waypoint[1]:.0f},{waypoint[2]:.0f})', 
                fontsize=8)
    
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_zlabel('Z Position (mm)')
    ax.set_title('3D End-Effector Trajectory')
    ax.legend()
    # Equal aspect ratio
    max_range = np.array([ee_positions_np[:, 0].max()-ee_positions_np[:, 0].min(),
                         ee_positions_np[:, 1].max()-ee_positions_np[:, 1].min(),
                         ee_positions_np[:, 2].max()-ee_positions_np[:, 2].min()]).max() / 2.0
    mid_x = (ee_positions_np[:, 0].max()+ee_positions_np[:, 0].min()) * 0.5
    mid_y = (ee_positions_np[:, 1].max()+ee_positions_np[:, 1].min()) * 0.5
    mid_z = (ee_positions_np[:, 2].max()+ee_positions_np[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

    # b) Joint velocities vs time (4 separate figures)
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'red', 'green', 'orange']

    for i in range(4):
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps_np, joint_velocities_np[:, i], line_styles[i], 
                color=colors[i], label=f"q̇{i+1} (Joint {i+1} Velocity)", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (deg/s)")
        plt.title(f"Joint {i+1} Velocity vs Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    # c) End-effector pose vs time
    plt.figure(figsize=(12, 8))
    line_styles = ['-', '--', '-.', ':']
    labels = ['x (mm)', 'y (mm)', 'z (mm)', 'pitch (deg)']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i in range(4):
        plt.plot(timestamps_np, ee_positions_np[:, i], line_styles[i], 
                color=colors[i], label=labels[i], linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position/Orientation')
    plt.title('End-Effector Pose vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # d) End-effector velocities vs time (using get_fwd_vel_kin data)
    plt.figure(figsize=(12, 8))
    line_styles = ['-', '--', '-.', ':']
    labels = ['ẋ (mm/s)', 'ẏ (mm/s)', 'ż (mm/s)', 'pitcḣ (deg/s)']
    colors = ['blue', 'red', 'green', 'orange']
    
    # Use actual velocities from get_fwd_vel_kin (first 4 components: x, y, z, pitch rates)
    for i in range(4):
        plt.plot(timestamps_np, ee_velocities_actual_np[:, i], line_styles[i], 
                color=colors[i], label=labels[i], linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('End-Effector Velocities vs Time (from get_fwd_vel_kin)')
    plt.grid(True)
    plt.legend()
    
    # Add horizontal line at 50 mm/s for reference
    plt.axhline(y=50, color='black', linestyle=':', alpha=0.7, label='Target Speed (50 mm/s)')
    plt.axhline(y=-50, color='black', linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Run data collection
    collect_data()
    # Plot data
    plot_data()