import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classes.Robot import Robot
from classes.TrajPlanner import TrajPlanner

def main():
    """Main function to execute joint-space cubic trajectory planning on the robot"""
    # Define task-space waypoints (x, y, z, alpha)
    ee_poses = [
        [25, -100, 150, -60],
        [150, 80, 300, 0],
        [250, -115, 75, -45],
        [25, -100, 150, -60]
    ]

    
    total_traj = 5  # seconds per waypoint
    n_intermediate = 998  # number of intermediate points between waypoints
    target_sample_rate = 200.0  # Hz
    timestamps = np.array([])
    waypoints_joint = []
    joint_angles = []
    ee_poses_recorded = []

    robot = Robot()

    # Lists to store data
    start_time = time.perf_counter()
    
    for wp in ee_poses: #converts from task space to joint space using inverse kinematics
        joint_wp = robot.get_ik(wp)
        waypoints_joint.append(joint_wp)
    waypoints_joint = np.array(waypoints_joint)

    print("\nComputed Joint-Space Waypoints (degrees):")
    for i, jp in enumerate(waypoints_joint):
        print(f"Waypoint {i+1}: {jp}")
    # Generate cubic trajectory in joint space
    planner = TrajPlanner(waypoints_joint) 
    trajectory = planner.get_cubic_traj(total_traj, n_intermediate) # generate cubic trajectory 
    print(trajectory)

    traj_time = .005
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    # Execute trajectory
    for i in range(trajectory.shape[0]):

        loop_start = time.perf_counter()
        # Command joint angles
        robot.write_joints(trajectory[i, 1:5])
        # Record data
        current_time = time.perf_counter() - start_time
        timestamps = np.append(timestamps, current_time)
        joint_angles = np.append(joint_angles, robot.get_joints_readings()[0, :])
        ee_poses_recorded = np.append(ee_poses_recorded, robot.get_ee_pos())
        # Maintain timing
        loop_end = time.perf_counter()
        elapsed = loop_end - loop_start
        desired_dt = 1.0 / target_sample_rate
        sleep_time = desired_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    timestamps_np =  np.array(timestamps)
    joint_angles_np = np.array(joint_angles)
    ee_positions_np = np.array(ee_poses_recorded)[:, 0:3]  # Extract x, y, z positions
    waypoints_np = np.array(ee_poses)

    # plot 3D plot showing path traced by end-effector in task space during entire trajectory, marking the 3 waypoints and their coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions_np[:, 0], ee_positions_np[:, 1], ee_positions_np[:, 2], 'b-', label='End-Effector Path')
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
        plt.plot(timestamps_np, joint_angles_np[:, i], label=f"Joint {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(f"Joint {i+1} Angle vs Time")
        plt.grid()
        plt.legend()
        plt.show()

    # Save data to pickle file
    filename = "lab4_2_data.pkl"
    save_data = {
        "timestamps_s": timestamps_np,
        "joint_deg": joint_angles_np,
        "ee_pos_mm": ee_positions_np,
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

if __name__ == "__main__":
    main()


    file_path = 'lab4_2_data.pkl'  
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
   

