import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classes.Robot import Robot


def main():
    traj_time = 5.0  # sec per waypoint
    target_sample_rate = 100  # Hz - aimed sampling rate
    
    # Define waypoints (x,y,z, alpha) for triangle in X-Z plane (keeping Y constant)
    waypoints = [
        [25, -100, 150, -60],           # Start/end position
        [150, 80, 300, 0],       # Waypoint 1
        [250, -115, 75, -45],        # Waypoint 2 
        [25, -100, 150, -60]            # Back to start
    ]
    



    print(f"\nConfiguration:")
    print(f"Trajectory time per waypoint: {traj_time} seconds")
    print(f"Target sampling rate: {target_sample_rate} Hz")
    print(f"Total waypoints: {len(waypoints)}")

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    # Lists to store data
    timestamps = []
    joint_angles = []
    ee_positions = []
    
    # For each waypoint

    for i in range(len(waypoints)):
        print(f"\nMoving to waypoint {i}: {waypoints[i]}")
        t0 = time.perf_counter()
        
        # Send command to next waypoint
        q = robot.get_ik(waypoints[i])
        robot.write_joints(q)
        
        # Record data until reaching target time
        while time.perf_counter() - t0 < traj_time:
            # Get current time and readings
            current_time = time.perf_counter() - t0 + (i * traj_time)
            current_readings = robot.get_joints_readings()
            current_ee_pos = robot.get_ee_pos()
            
            # Store data
            timestamps.append(current_time)
            joint_angles.append(current_readings[0])  # First row contains joint angles
            ee_positions.append(current_ee_pos[:4])  # Store only all 4 components (x,y,z,alpha)
            
            # Try to maintain target sample rate
            remaining_time = (1.0 / target_sample_rate) - (time.perf_counter() - (t0 + current_time - i * traj_time))
            if remaining_time > 0:
                time.sleep(remaining_time * 0.8)
            
            # # Print status every ~1 second
            # if len(timestamps) % target_sample_rate == 0:
            #     _print_readings(current_readings, current_ee_pos)

    # Convert lists to numpy arrays
    timestamps_np = np.array(timestamps)
    joint_angles_np = np.array(joint_angles)
    ee_positions_np = np.array(ee_positions)
    waypoints_np = np.array(waypoints)

    #plots:
    



    # Plot end-effector pose vs time. plot 4 lines showing x, y, z (mm) and alpha (deg) versus time (sec). Use distinguishable line styles and include a legend. Seperate plots for each x, y, z, alpha
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # X position subplot
    ax1.plot(timestamps_np, ee_positions_np[:, 0], 'b-', linewidth=2)
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('X Position (mm)')
    ax1.set_title('End-Effector X Position vs Time')
    ax1.grid(True)

    # Y position subplot  
    ax2.plot(timestamps_np, ee_positions_np[:, 1], 'g-', linewidth=2)
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title('End-Effector Y Position vs Time')
    ax2.grid(True)

    # Z position subplot
    ax3.plot(timestamps_np, ee_positions_np[:, 2], 'r-', linewidth=2)
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Z Position (mm)')
    ax3.set_title('End-Effector Z Position vs Time')
    ax3.grid(True)

    # Pitch angle subplot
    ax4.plot(timestamps_np, ee_positions_np[:, 3], 'm-', linewidth=2)
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Pitch Angle (deg)')
    ax4.set_title('End-Effector Pitch Angle vs Time')
    ax4.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    


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
    filename = "lab3_3_data.pkl"
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

    # Shutdown
    robot.close()

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
