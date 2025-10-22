# python ee471-codebase/lab4_starter.py



import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from classes.Robot import Robot
from classes.TrajPlanner import TrajPlanner

def collect_data():
    """
    Collects data for the robot's movement and saves it to a pickle file.
    """
    traj_time = 5  # Trajectory time per segment
    points_num = 998  # Number of intermediate waypoints per segment
    robot = Robot()

    # Define task-space setpoints
    waypoints_np = np.array([
        [25, -100, 150, -60],
        [150, 80, 300, 0],
        [250, -115, 75, -45],
        [25, -100, 150, -60]  # Return to start
    ])

    planner = TrajPlanner(waypoints_np)
    trajectories = planner.get_quintic_traj(traj_time, points_num)

    # Convert task-space trajectory to joint-space trajectory using IK
    joint_trajectories = []
    for i in range(len(trajectories)):
        t = trajectories[i, 0]
        x, y, z, phi = trajectories[i, 1:]
        q = robot.get_ik([x, y, z, phi])
        joint_trajectories.append([t, q[0], q[1], q[2], q[3]])
    trajectories = np.array(joint_trajectories)




    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Total trajectory time: {trajectories[-1, 0]:.2f} seconds")

    # Calculate time step between trajectory points
    time_step = trajectories[1, 0] - trajectories[0, 0]
    print(f"Time step between points: {time_step*1000:.2f} ms")
    print(f"Command frequency: {1/time_step:.1f} Hz")

    # Pre-allocate data (over-allocate for safety during continuous sampling)
    total_points = len(trajectories)
    max_samples = total_points * 5  # Assume we might sample 5x during execution
    data_time = np.zeros(max_samples)
    data_ee_poses = np.zeros((max_samples, 4))
    data_q = np.zeros((max_samples, 4))
    count = 0

    # Initialize robot
    print("\nInitializing robot...")
    robot.write_motor_state(True)
    # robot.write_profile_acceleration(20)

    # Move to starting position
    print("Moving to start position...")
    robot.write_time(traj_time)
    robot.write_joints(trajectories[0, 1:])
    time.sleep(traj_time)  # Wait for completion

    print("\nExecuting trajectory...")
    robot.write_time(time_step)
    start_time = time.perf_counter()

    # Execute trajectory by streaming commands
    for i in range(1, len(trajectories)):
        # Calculate when this command should be sent
        target_time = start_time + trajectories[i, 0]
        
        # Wait until it's time to send this command
        while time.perf_counter() < target_time:
            # Collect data while waiting
            current_time = time.perf_counter() - start_time
            
            if count < max_samples:
                data_q[count, :] = robot.get_joints_readings()[0, :]
                data_time[count] = current_time
                data_ee_poses[count, :] = robot.get_ee_pos(data_q[count, :])[0:4]
                count += 1
            # Small sleep to preventfget CPU overload
            time.sleep(0.001)  # 1ms sleep
        
        # Send the command at the scheduled time
        robot.write_joints(trajectories[i, 1:])
    
    total_time = time.perf_counter() - start_time
    print(f"\nTrajectory complete!")
    print(f"Planned time: {trajectories[-1, 0]:.2f}s")
    print(f"Actual time: {total_time:.2f}s")
    print(f"Total samples collected: {count}")
    print(f"Average sample rate: {count/total_time:.1f} Hz")

    # Trim unused space
    timestamps_np = data_time[:count]
    data_ee_poses = data_ee_poses[:count, :]
    data_q = data_q[:count, :]

    # Save data to a picke file (TODO)


    


    
    # plot 3D plot showing path traced by end-effector in task space during entire trajectory, marking the 3 waypoints and their coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data_ee_poses[:, 0], data_ee_poses[:, 1], data_ee_poses[:, 2], 'b-', label='End-Effector Path')
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

    # Save data to pickle file
    filename = "lab4_3_data.pkl"
    save_data = {
        "timestamps_s": timestamps_np,
        "joint_deg": q,
        "ee_pos_mm": data_ee_poses,
        "waypoints_deg": waypoints_np,
        "traj_time_s": traj_time
    }
    
    with open(filename, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nData saved to {filename}")



def plot_from_pickle(filename):
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


    
    robot.close()


# def plot_data():
#     """
#     Loads data from a pickle file and plots it.
#     (TODO)
#     """


    
if __name__ == "__main__":
    
    # # Collect data
    collect_data()
    plot_from_pickle("lab4_3_data.pkl")
    # Plot data
    # plot_data()
