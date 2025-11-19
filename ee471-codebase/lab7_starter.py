"""
(c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
Lab 7 Starter Code: Position-Based Visual Servoing with PID Control
Implements real-time visual servoing to track an AprilTag target.
"""

import numpy as np
import cv2
import time
import pickle

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.AprilTags import AprilTags

class PIDController:
    """
    PID controller for position-based visual servoing.
    """
    def __init__(self, dim=3, dt=0.05):
        """
        Initialize PID controller.
        
        Args:
            dim: Dimension of control (3 for x, y, z)
            dt: Control timestep in seconds (MUST match control loop timing!)
        """
        # =============================================================================
        # TODO: Initialize PID gains
        # Hint: Start with small values and tune accordingly
        # YOUR CODE HERE
        self.Kp = np.eye(dim) * 1.2 # Replace with gain matrix (dim x dim)
        self.Ki = np.eye(dim) * 0.01 # Replace with gain matrix (dim x dim)
        self.Kd = np.eye(dim) * 0.01  # Replace with gain matrix (dim x dim)


        # Initialize error tracking variables
        self.error_integral = np.zeros(dim)
        self.error_prev = np.zeros(dim)
        self.dt = dt
        
    def compute(self, error):
        """
        Compute PID control output.
        
        Args:
            error: Position error vector (mm)
            
        Returns:
            control_output: Velocity command (mm/s)
        """
        # =============================================================================
        # TODO: Implement PID control computation (from Pre-Lab 7)
        # Hint: output = Kp*error + Ki*integral + Kd*derivative
        # Don't forget to update error_integral and error_prev!

        # Proportional
        P = self.Kp @ error

        # Integral
        self.error_integral += error * self.dt
        I = self.Ki @ self.error_integral

        # Derivative 
        D = self.Kd @ ((error - self.error_prev) / self.dt)
        self.error_prev = error

        # PID output
        output = P + I + D

        # Store last PID terms
        # self.last_P, self.last_I, self.last_D = P, I, D
        # self.error_prev = error.copy() 
        
        return output


def main():
    """
    Main visual servoing control loop.
    """
    print("="*60)
    print("Lab 7: Position-Based Visual Servoing")
    print("="*60)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    print("\nInitializing system...")
    
    max_samples = 10000
    count = 0  # Sample counter
    data_time = np.zeros(max_samples)
    data_pid = np.zeros((max_samples, 3))           # PID output velocity commands
    data_joint_vel = np.zeros((max_samples, 4))     # Joint velocities 
    pos_current = np.zeros((max_samples, 3))        # Current EE positions
    pos_desired = np.zeros((max_samples, 3))        # Desired EE positions
    data_error = np.zeros((max_samples, 3))      # Position errors
    tag_detected_list = []                          # Tag detection status


    # Initialize robot, camera, and AprilTag detector
    # Hint: Create Robot(), Realsense(), and AprilTags() instances
    # YOUR CODE HERE
    robot = Robot()      # Replace with Robot()
    camera = Realsense()    # Replace with Realsense()
    detector = AprilTags()   # Replace with AprilTags()
    
    # Get camera intrinsics
    # Hint: Use camera.get_intrinsics()
    # YOUR CODE HERE
    intrinsics = camera.get_intrinsics()  # Replace with actual intrinsics
    
    
    # TODO: Define control timestep (CRITICAL - must match PID dt!)
    # Hint: 0.05 seconds = 20 Hz, 0.02 seconds = 50 Hz
    # YOUR CODE HERE
    dt = 0.05  # Control loop period in seconds
    pid = PIDController(dim=3, dt=dt)   #initialize pid
    
    # Set AprilTag physical size in millimeters
    # Hint: Measure your actual tag!
    TAG_SIZE = 40.0  # Update this value
    
    
    # TODO: Initialize PID controller with SAME dt as control loop
    # Hint: pid = PIDController(dim=3, dt=dt)
    pid = PIDController(dim=3, dt=dt)  # Replace with PIDController instance
    
    
    # Load camera-robot calibration matrix
    # Hint: Use np.load('camera_robot_transform.npy')
    # YOUR CODE HERE
    try:
        T_cam_to_robot = np.load('camera_robot_transform.npy')
        print("Calibration loaded successfully")
    except FileNotFoundError:
        print("Error: camera_robot_transform.npy not found!")
        print("Please run lab6_2.py first to calibrate.")
        return
    
    
    # Define desired offset from tag (mm)
    # Hint: This defines where end-effector should be relative to tag
    # Example: [0, 0, 50] means 50mm above tag
    # YOUR CODE HERE
    target_offset = np.array([-40, 0, 75])  # Adjust as needed
    print(f"Target offset from tag: {target_offset} mm")
    
    
    # =========================================================================
    # MOVE TO START POSITION
    # =========================================================================
    print("\nMoving to start position...")
    
    # Set position mode and trajectory time
    robot.write_mode("position")
    traj_time = 3.0
    robot.write_time(traj_time)
    
    
    # Start position: [x, y, z, gripper_angle] in mm and degrees
    start_position = [100, 0, 220, -15]
    start_joints = robot.get_ik(start_position)

    # Move to start position: write joints and wait for motion to complete
    # Hint: Use robot.get_ik(), robot.write_joints(), time.sleep()
    robot.write_time(5.0)
    robot.write_joints(start_joints)
    time.sleep(5.0)
    # YOUR CODE HERE
    data_start_time = time.perf_counter()  # Move this before the loop
    
    # Switch robot to velocity control mode
    # Hint: Use robot.write_mode("velocity")
    # YOUR CODE HERE
    robot.write_mode("velocity")
    
    print("Robot ready for visual servoing")
    
    
    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    print("\nStarting visual servoing control loop...")
    print("Press 'q' to quit\n")
    
    iteration = 0
    
    try:
        while True:
            # Record start time for fixed timestep enforcement
            # Hint: start_time = time.time()
            # YOUR CODE HERE
            start_time = time.time()  # Replace with actual time
            
            
            # -----------------------------------------------------------------
            # STEP 1: CAPTURE FRAME AND DETECT TAG
            # -----------------------------------------------------------------
            
            # Get camera frame
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                continue
            
            # Detect AprilTags in frame
            # Hint: Use detector.detect_tags(color_frame)
            tags = detector.detect_tags(color_frame)  # Replace with detected tags
            
            
            # Check if any tags detected
            if len(tags) > 0:
                tag = tags[0]  # Use first detected tag
                
                # Draw tag detection on image for visualization
                color_frame = detector.draw_tags(color_frame, tag)
                
                # -----------------------------------------------------------------
                # STEP 2: GET TAG POSE IN CAMERA FRAME
                # -----------------------------------------------------------------
                
                # Get tag pose using PnP
                # Hint: detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                # Returns: (rotation_matrix, translation_vector)
                # YOUR CODE HERE
                rot_matrix = detector.get_tag_pose(tag.corners, intrinsics,TAG_SIZE)[0]
                trans_vector = detector.get_tag_pose(tag.corners,intrinsics,TAG_SIZE)[1]   # Replace with actual rotation and translation
                
                
                # Extract tag position in camera frame (already in mm)
                # Hint: Flatten trans_vector to get 1D array
                # YOUR CODE HERE
                tag_pos_camera  = trans_vector.flatten()  # Replace with tag position in camera frame (shape: 3,)
                
                
                # -----------------------------------------------------------------
                # STEP 3: TRANSFORM TO ROBOT FRAME
                # -----------------------------------------------------------------
                
                # TODO: Convert to homogeneous coordinates
                # Hint: Append 1 to make [x, y, z, 1]
                tag_pos_camera_hom = tag_pos_camera.tolist() + [1]  # Replace with homogeneous coordinates (shape: 4,)
                
                # TODO: Apply camera-to-robot transformation
                tag_pos_robot_hom = T_cam_to_robot @ tag_pos_camera_hom  # Replace with tag position in robot frame (homogeneous, shape: 4,)

                # Extract 3D position from homogeneous coordinates
                # Hint: Take first 3 elements
                tag_pos_robot = tag_pos_robot_hom[:3]  # Replace with tag position in robot frame (shape: 3,)
                
                
                # -----------------------------------------------------------------
                # STEP 4: CALCULATE DESIRED END-EFFECTOR POSITION
                # -----------------------------------------------------------------
                
                # Add offset to tag position to get desired EE position
                # Hint: desired_ee_pos = tag_pos_robot + target_offset
                # YOUR CODE HERE
                desired_ee_pos = tag_pos_robot + target_offset  # Replace with desired position
                
                
                # -----------------------------------------------------------------
                # STEP 5: GET CURRENT END-EFFECTOR POSITION
                # -----------------------------------------------------------------
                
                # Get current joint positions
                # YOUR CODE HERE
                current_joints = robot.get_joints_readings()[0]  # Replace with current joint angles
                
                
                # Get current end-effector position using forward kinematics
                # Hint: robot.get_ee_pos(current_joints) returns [x, y, z, ...]
                # Take only first 3 elements (position)
                # YOUR CODE HERE
                current_ee_pos = robot.get_ee_pos()[:3]  # Replace with current EE position (shape: 3,)
                
                
                # -----------------------------------------------------------------
                # STEP 6: CALCULATE POSITION ERROR
                # -----------------------------------------------------------------
                
                # Compute position error
                # Hint: error = desired_position - current_position
                # YOUR CODE HERE
                error = desired_ee_pos - current_ee_pos  # Replace with error vector
                
                
                # -----------------------------------------------------------------
                # STEP 7: COMPUTE PID CONTROL OUTPUT
                # -----------------------------------------------------------------
                
                #  Use PID controller to compute velocity command
                # Hint: Call pid.compute()
                # YOUR CODE HERE
                velocity_cmd = pid.compute(error)  # Replace with velocity command (shape: 3,)
                
                
                # -----------------------------------------------------------------
                # STEP 8: CONVERT TO JOINT VELOCITIES
                # -----------------------------------------------------------------
                
                # Get robot Jacobian at current configuration
                # Hint: robot.get_jacobian()
                # YOUR CODE HERE
                J = robot.get_jacobian(current_joints[0])  # Replace with Jacobian matrix
                
                
                # TODO: Extract position part of Jacobian (first 3 rows)
                # Hint: J_linear = J[:3, :]
                # YOUR CODE HERE
                J_linear = J[:3, :]  # Replace with position Jacobian
                
                
                # TODO: Compute joint velocities using pseudo-inverse
                # Hint: joint_vel = pinv(J_linear) @ velocity_cmd
                # Use np.linalg.pinv()
                # YOUR CODE HERE
                joint_vel = np.linalg.pinv(J_linear) @ velocity_cmd  # Replace with joint velocities
                
                
                # -----------------------------------------------------------------
                # STEP 9: COMMAND ROBOT
                # -----------------------------------------------------------------
                
                # TODO: Send joint velocities to robot
                # Hint: robot.write_velocities() in degrees
                # OpenManipulator-X has 4 joints (excluding gripper)
                # YOUR CODE HERE
                robot.write_velocities(np.rad2deg(joint_vel))  # Replace with command to send joint velocities
                
                
                # -----------------------------------------------------------------
                # STEP 10: DISPLAY STATUS
                # -----------------------------------------------------------------



                if count < max_samples:
                    data_time[count] = time.perf_counter() - data_start_time
                    data_pid[count, :] = velocity_cmd
                    data_joint_vel[count, :] = np.rad2deg(joint_vel)
                    pos_current[count, :] = current_ee_pos
                    pos_desired[count, :] = desired_ee_pos
                    data_error[count, :] = error
                    if len(tags) > 0:
                        tag_detected_list.append(True)
                    else:
                        tag_detected_list.append(False)
                    count += 1

                # Print status every 40 iterations (~2 seconds at 20Hz)
                if iteration % 40 == 0:
                    print(f"\nIteration: {iteration}")
                    print(f"Tag position (robot): {tag_pos_robot}")
                    print(f"Current EE position:  {current_ee_pos}")
                    print(f"Desired EE position:  {desired_ee_pos}")
                    print(f"Error: {error} mm")
                    print(f"Error magnitude: {np.linalg.norm(error):.2f} mm")
                
            else:
                # -----------------------------------------------------------------
                # NO TAG DETECTED - STOP ROBOT
                # -----------------------------------------------------------------
                
                # TODO: Stop robot motion by sending zero velocities
                # Hint: robot.write_velocities([0, 0, 0, 0])
                robot.write_velocities([0, 0, 0, 0])
                
                
                if iteration % 40 == 0:
                    print("\nNo AprilTag detected - robot stopped")
            
            
            # -----------------------------------------------------------------
            # DISPLAY AND USER INTERACTION
            # -----------------------------------------------------------------
            
            # Display camera image
            cv2.imshow('Visual Servoing', color_frame)
            key = cv2.waitKey(1)
            
            # Check for quit key press ('q' or ESC)
            if key & 0xFF == ord('q') or key == 27:
                print("\nQuitting...")
                break
            
            iteration += 1
            
            
            # -----------------------------------------------------------------
            # MAINTAIN FIXED TIMESTEP (CRITICAL!)
            # -----------------------------------------------------------------
            
            # Enforce consistent loop timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # =====================================================================
        # CLEANUP
        # =====================================================================
        # Save data before cleanup
            # Create and save data dictionary
        data = {
            'time': data_time, # Timestamp (s)
            'pos_current': pos_current, # Current EE position [x,y,z] (mm)
            'pos_desired': pos_desired, # Desired EE position [x,y,z] (mm)
            'error': data_error, # Position error [ex,ey,ez] (mm)
            'control_output': data_pid, # PID output [vx,vy,vz] (mm/s)
            'joint_vel': data_joint_vel, # Joint velocities (rad/s)
            'tag_detected': tag_detected_list  # Boolean: tag visible
        }
            
        filename = 'lab7_data.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved! Collected {count} samples")
        
        # Then do the cleanup
        print("\nStopping robot and cleaning up...")
        plot_data('lab7_data.pkl')
        robot.write_velocities([0, 0, 0, 0])
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")

        



def plot_data(filename='lab7_data.pkl'):
    """
    Load data and create required plots per lab requirements.
    Trims unused zero rows from preallocated arrays so plots look clean.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Load data
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return

    # Extract raw arrays
    time_data = np.array(data['time'])
    pos_current = np.array(data['pos_current'])
    pos_desired = np.array(data['pos_desired'])
    error_data = np.array(data['error'])
    control_output = np.array(data['control_output'])
    joint_vel = np.array(data['joint_vel'])
    tag_detected = np.array(data['tag_detected'], dtype=bool)  # list -> array

    # -------------------------------------------------------------------------
    # Trim to only valid samples (non-zero positions OR positive time)
    # -------------------------------------------------------------------------
    # A row is "valid" if either pos_current or pos_desired is nonzero
    pos_nonzero = np.any(pos_current != 0, axis=1) | np.any(pos_desired != 0, axis=1)
    # Also accept any row with time > 0 (in case your very first sample has zeros)
    time_nonzero = time_data > 0

    valid_mask = pos_nonzero | time_nonzero
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("plot_data: No valid samples found (all zeros).")
        return

    start = valid_indices[0]
    end = valid_indices[-1] + 1  # slice end is exclusive

    # Slice everything consistently
    time_data = time_data[start:end]
    pos_current = pos_current[start:end, :]
    pos_desired = pos_desired[start:end, :]
    error_data = error_data[start:end, :]
    control_output = control_output[start:end, :]
    joint_vel = joint_vel[start:end, :]

    # If tag_detected is shorter (it's a list with length=count), align it
    if tag_detected.size >= end:
        tag_detected = tag_detected[start:end]
    else:
        # Just truncate/match to time length
        tag_detected = tag_detected[: time_data.shape[0]]

    # -------------------------------------------------------------------------
    # 1. POSITION TRACKING - 3 subplots (X, Y, Z)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        axes1[i].plot(time_data, pos_desired[:, i], 'r--', label='Desired', linewidth=2)
        axes1[i].plot(time_data, pos_current[:, i], 'b-', label='Actual', linewidth=1)
        axes1[i].set_ylabel('Position (mm)')
        axes1[i].set_title(f'{axes_labels[i]} Position Tracking')
        axes1[i].legend()
        axes1[i].grid(True)

    axes1[-1].set_xlabel('Time (s)')
    fig1.suptitle('Position Tracking: Desired vs. Actual Position', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 2. ERROR ANALYSIS - 3 subplots
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        axes2[i].plot(time_data, error_data[:, i], 'r-', linewidth=1, label='Position Error')
        axes2[i].axhline(y=2, color='g', linestyle='--', alpha=0.7, label='+2mm threshold')
        axes2[i].axhline(y=-2, color='g', linestyle='--', alpha=0.7, label='-2mm threshold')

        rms_error = np.sqrt(np.mean(error_data[:, i] ** 2))
        axes2[i].text(
            0.02, 0.98, f'RMS: {rms_error:.2f}mm',
            transform=axes2[i].transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        axes2[i].set_xlabel('Time (s)')
        axes2[i].set_ylabel(f'{axes_labels[i]} Error (mm)')
        axes2[i].set_title(f'{axes_labels[i]} Position Error')
        axes2[i].legend()
        axes2[i].grid(True)

    fig2.suptitle('Position Error Analysis: X, Y, Z Components', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 3. CONTROL OUTPUT - 3 subplots
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        axes3[i].plot(time_data, control_output[:, i], 'b-', linewidth=1)
        axes3[i].set_xlabel('Time (s)')
        axes3[i].set_ylabel(f'{axes_labels[i]} Velocity Command (mm/s)')
        axes3[i].set_title(f'{axes_labels[i]} PID Output')
        axes3[i].grid(True)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 4. JOINT VELOCITIES - single figure
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
    colors = ['blue', 'red', 'green', 'orange']

    for i in range(min(4, joint_vel.shape[1])):
        ax4.plot(time_data, joint_vel[:, i], label=joint_labels[i],
                 color=colors[i], linewidth=1)

    # Velocity "limits" line (note: your data is in deg/s here)
    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7,
                label='±0.5 (label only, check units)')
    ax4.axhline(y=-0.5, color='black', linestyle='--', alpha=0.7)

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Joint Velocity (deg/s)')
    ax4.set_title('Joint Velocities vs Time')
    ax4.legend()
    ax4.grid(True)
    plt.show()

    # -------------------------------------------------------------------------
    # 5. 3D TRAJECTORY - single figure
    # -------------------------------------------------------------------------
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')

    # Filter again in case some leading points are still all-zero
    valid_current = pos_current[np.any(pos_current != 0, axis=1)]
    valid_desired = pos_desired[np.any(pos_desired != 0, axis=1)]

    if len(valid_current) > 1:
        ax5.plot(valid_current[:, 0], valid_current[:, 1], valid_current[:, 2],
                 'b-', linewidth=3, label='Actual Path', alpha=0.8)

        ax5.scatter(valid_current[0, 0], valid_current[0, 1], valid_current[0, 2],
                    color='green', s=150, label='Start', marker='o', edgecolor='black')
        ax5.scatter(valid_current[-1, 0], valid_current[-1, 1], valid_current[-1, 2],
                    color='red', s=150, label='End', marker='s', edgecolor='black')

    if len(valid_desired) > 1:
        ax5.plot(valid_desired[:, 0], valid_desired[:, 1], valid_desired[:, 2],
                 'r--', linewidth=2, label='Desired Path', alpha=0.7)

    if len(valid_current) > 0:
        all_points = valid_current
        if len(valid_desired) > 0:
            all_points = np.vstack([valid_current, valid_desired])

        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax5.set_xlim(mid_x - max_range, mid_x + max_range)
        ax5.set_ylim(mid_y - max_range, mid_y + max_range)
        ax5.set_zlim(mid_z - max_range, mid_z + max_range)

    ax5.set_xlabel('X Position (mm)')
    ax5.set_ylabel('Y Position (mm)')
    ax5.set_zlabel('Z Position (mm)')
    ax5.set_title('3D End-Effector Trajectory')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()




    

    
if __name__ == "__main__":
    main()