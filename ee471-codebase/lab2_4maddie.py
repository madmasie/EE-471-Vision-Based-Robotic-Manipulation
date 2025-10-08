import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classes.Robot import Robot

def _print_readings(readings, ee_pos):
    # Accepts a 3xN array-like: [deg; deg/s; mA]
    q_deg, qd_dps, I_mA = readings
    q_str  = ",".join(f"{x:5.1f}" for x in q_deg)
    qd_str = ",".join(f"{x:5.1f}" for x in qd_dps)
    I_str  = ",".join(f"{x:5.0f}" for x in I_mA)
    ee_str = ",".join(f"{x:5.1f}" for x in ee_pos[:3])  # Only position, not orientation
    print(f"q(deg): [{q_str}] | qdot(deg/s): [{qd_str}] | I(mA): [{I_str}] | ee(mm): [{ee_str}]")

def main():
    traj_time = 5.0  # sec per waypoint
    target_sample_rate = 100  # Hz - aimed sampling rate
    
    # Define waypoints for triangle in X-Z plane (keeping Y constant)
    waypoints = [
        [0, -45, 60, 50],           # Start/end position
        [0, 10, 50, -45],       # Waypoint 1
        [0, 10, 0, -80],        # Waypoint 2 
        [0, -45, 60, 50]            # Back to start
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
    for i in range(len(waypoints)-1):
        print(f"\nMoving to waypoint {i+1}: {waypoints[i+1]}")
        t0 = time.perf_counter()
        
        # Send command to next waypoint
        robot.write_joints(waypoints[i+1])
        
        # Record data until reaching target time
        while time.perf_counter() - t0 < traj_time:
            # Get current time and readings
            current_time = time.perf_counter() - t0 + (i * traj_time)
            current_readings = robot.get_joints_readings()
            current_ee_pos = robot.get_ee_pos()
            
            # Store data
            timestamps.append(current_time)
            joint_angles.append(current_readings[0])  # First row contains joint angles
            ee_positions.append(current_ee_pos[:3])  # Store only x,y,z
            
            # Try to maintain target sample rate
            remaining_time = (1.0 / target_sample_rate) - (time.perf_counter() - (t0 + current_time - i * traj_time))
            if remaining_time > 0:
                time.sleep(remaining_time * 0.8)
            
            # Print status every ~1 second
            if len(timestamps) % target_sample_rate == 0:
                _print_readings(current_readings, current_ee_pos)

    # Convert lists to numpy arrays
    timestamps_np = np.array(timestamps)
    joint_angles_np = np.array(joint_angles)
    ee_positions_np = np.array(ee_positions)
    waypoints_np = np.array(waypoints)

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
    
    # Extract x,z coordinates of waypoints (converting joint angles to end-effector positions)
    waypoint_ee_pos = []
    for waypoint in waypoints:
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
    plt.plot(waypoint_ee_pos[:, 0], [T[1,3] for T in [robot.get_fk(w) for w in waypoints]], 'ro', label='Waypoints')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('End-Effector X-Y Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Save data to pickle file
    filename = "lab2_4_data.pkl"
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
