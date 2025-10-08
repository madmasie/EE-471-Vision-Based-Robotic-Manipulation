# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import time
import matplotlib.pyplot as plt
import numpy
from classes.Robot import Robot
import pickle

"""
This script demonstrates basic operation of the OpenManipulator-X via the Robot class.
It initializes the robot, sets a time-based move profile, moves the base joint
through a few waypoints while printing live joint readings, and toggles the gripper.
"""

def _print_readings(readings):
    # Accepts a 3xN array-like: [deg; deg/s; mA]
    q_deg, qd_dps, I_mA = readings
    q_str  = ",".join(f"{x:5.1f}" for x in q_deg)
    qd_str = ",".join(f"{x:5.1f}" for x in qd_dps)
    I_str  = ",".join(f"{x:5.0f}" for x in I_mA)
    print(f"q(deg): [{q_str}] | qdot(deg/s): [{qd_str}] | I(mA): [{I_str}]")

def main():
    traj_time = 2.0                 # sec
    target_sample_rate = 100  # Hz - aimed sampling rate
    joint_angles = []
    timestamps = []
    
    print(f"\nConfiguration:")
    print(f"Trajectory time: {traj_time} seconds")
    print(f"Target sampling rate: {target_sample_rate} Hz")
    print(f"Expected samples: {int(traj_time * target_sample_rate)}")

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    #send the arm to its zero pose.
    print("Homing to [0, 0, 0, 0] deg ...")
    robot.write_joints([0, 0, 0, 0])
    time.sleep(traj_time)


    # Target joint angles
    target_angles = [45, -30, 30, 75]
    
    print(f"\nStarting motion tracking...")
    t0 = time.perf_counter()
    
    # Record data for the full trajectory duration
    while time.perf_counter() - t0 < traj_time:
        # Get current time and readings
        current_time = time.perf_counter() - t0
        current_readings = robot.get_joints_readings()
        
        # Only command motion once at the start
        if len(timestamps) == 0:
            print(f"Moving to {target_angles} deg...")
            robot.write_joints(target_angles)
            
        # Store data
        timestamps.append(current_time)
        joint_angles.append(current_readings[0])
        
        # Try to maintain target sample rate (avoid busy waiting)
        remaining_time = (1.0 / target_sample_rate) - (time.perf_counter() - (t0 + current_time))
        if remaining_time > 0:
            time.sleep(remaining_time * 0.8)  # Sleep for slightly less than full interval to account for overhead
        
        # Print status every ~1 second
        if len(timestamps) % 10 == 0:  # Adjust frequency based on actual sampling rate
            _print_readings(current_readings)

    #convert joint_angles and timestamps to numpy arrays
    joint_angles_np = numpy.array(joint_angles) #shape (N, 4)
    timestamps_np = numpy.array(timestamps) #shape (N,)

    #coverting target_angles to a numpy array for saving later
    target_angles = numpy.array(target_angles) #shape (4,)

    # Create figure with 4 subplots, one per joint
    plt.figure(figsize=(12, 10))
    
    # Add space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Joint 1
    plt.subplot(4, 1, 1)
    plt.plot(timestamps_np, joint_angles_np[:, 0], label="Joint 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Joint 1 Angle vs Time")
    plt.grid()
    plt.legend()

    # Joint 2
    plt.subplot(4, 1, 2)
    plt.plot(timestamps_np, joint_angles_np[:, 1], label="Joint 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Joint 2 Angle vs Time")
    plt.grid()
    plt.legend()

    # Joint 3
    plt.subplot(4, 1, 3)
    plt.plot(timestamps_np, joint_angles_np[:, 2], label="Joint 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Joint 3 Angle vs Time")
    plt.grid()
    plt.legend()

    # Joint 4
    plt.subplot(4, 1, 4)
    plt.plot(timestamps_np, joint_angles_np[:, 3], label="Joint 4")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Joint 4 Angle vs Time")
    plt.grid()
    plt.legend()

    # Display the plot
    plt.show()


    # Timing histogram. Compute ∆t using np.diff(t) and plot a histogram of the sampling intervals
    delta_t = numpy.diff(timestamps_np)
    
    print(f"Number of samples: {len(timestamps_np)}")
    print(f"Total time span: {timestamps_np[-1] - timestamps_np[0]:.4f} s")
    
    #computing timing statistics
    mean_t = numpy.mean(delta_t)
    median_t = numpy.median(delta_t)
    min_t = numpy.min(delta_t)
    max_t = numpy.max(delta_t)
    std_t = numpy.std(delta_t)
    
    print(f"\nTiming Statistics:")
    print(f"Mean delta_t: {mean_t:.4f} s")
    print(f"Median delta_t: {median_t:.4f} s")
    print(f"Min delta_t: {min_t:.4f} s")
    print(f"Max delta_t: {max_t:.4f} s")
    print(f"Standard deviation of delta_t: {std_t:.4f} s")
    
    # Plot histogram of sampling intervals
    plt.figure(figsize=(8, 6))
    plt.hist(delta_t, bins=30, edgecolor='black')
    plt.xlabel('Time Interval (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sampling Intervals')
    plt.grid(True)
    plt.show()



    #Helper functions to save data to a pickle file
    def save_to_pickle(data: dict, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    def load_from_pickle(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)

    #Loads the dict. Recreates the four joint time-series subplots from timestamps_s and joint_deg. Recomputes and plots the ∆t histogram and prints the same statistics as in Part 2
    def plot_from_pickle(filename):
        # Load data from pickle file
        data = load_from_pickle(filename)
        timestamps_np = data["timestamps_s"]
        joint_angles_np = data["joint_deg"]

        # Plot joint angles
        plt.figure(figsize=(12, 10))
        plt.subplots_adjust(hspace=0.4)
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(timestamps_np, joint_angles_np[:, i], label=f"Joint {i+1}")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (deg)")
            plt.title(f"Joint {i+1} Angle vs Time")
            plt.grid()
            plt.legend()
        plt.show()

        # Timing histogram
        delta_t = numpy.diff(timestamps_np)
        print(f"Number of samples: {len(timestamps_np)}")
        print(f"Total time span: {timestamps_np[-1] - timestamps_np[0]:.4f} s")
        mean_t = numpy.mean(delta_t)
        median_t = numpy.median(delta_t)
        min_t = numpy.min(delta_t)
        max_t = numpy.max(delta_t)
        std_t = numpy.std(delta_t)
        print(f"\nTiming Statistics:")
        print(f"Mean delta_t: {mean_t:.4f} s")
        print(f"Median delta_t: {median_t:.4f} s")
        print(f"Min delta_t: {min_t:.4f} s")
        print(f"Max delta_t: {max_t:.4f} s")
        print(f"Standard deviation of delta_t: {std_t:.4f} s")
        plt.figure(figsize=(8, 6))
        plt.hist(delta_t, bins=30, edgecolor='black')
        plt.xlabel('Time Interval (s)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Sampling Intervals')
        plt.grid(True)
        plt.show()
        
        

        
        

    #10s
    filename = "lab1_data_2s.pkl"
    # Save
    save_to_pickle({"timestamps_s": timestamps_np, "joint_deg": joint_angles_np, "target_deg":
        target_angles, "traj_time_s": 2.0}, filename)
    # Load
    data = load_from_pickle(filename)
    t = data["timestamps_s"]; Q = data["joint_deg"]
    plot_from_pickle(filename)  #calling plot_from_pickle


    


    # Shutdown
    robot.close()

if __name__ == "__main__":
    main()
