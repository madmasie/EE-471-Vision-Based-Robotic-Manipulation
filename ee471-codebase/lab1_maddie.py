# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import time
import matplotlib.pyplot as plt
import numpy
from classes.Robot import Robot

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
    traj_time = 10.0                 # sec
    joint_angles = []
    timestamps = []

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    #send the arm to its zero pose.
    print("Homing to [0, 0, 0, 0] deg ...")
    robot.write_joints([0, 0, 0, 0])
    time.sleep(traj_time)


    t0 = time.perf_counter()
    while time.perf_counter() - t0 < traj_time:
        _print_readings(robot.get_joints_readings())
        #moving to [45, -30, 30, 75]
        print("moving to [45, -30, 30, 75] deg ...")
        robot.write_joints([45, -30, 30, 75])
        
        timestamps.append(time.perf_counter() - t0)
        joint_angles.append(robot.get_joints_readings()[0])

    #convert joint_angles and timestamps to numpy arrays
    joint_angles_np = numpy.array(joint_angles) #shape (N, 4)
    timestamps_np = numpy.array(timestamps) #shape (N,)

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



    #mean, median, min, max, standard deviation histogeam of delta_ta uing np.diff(t)
    mean_t = numpy.mean(numpy.diff(timestamps_np))
    median_t = numpy.median(numpy.diff(timestamps_np))
    min_t = numpy.min(numpy.diff(timestamps_np))
    max_t = numpy.max(numpy.diff(timestamps_np))
    std_t = numpy.std(numpy.diff(timestamps_np))
    print(f"Mean delta_t: {mean_t:.4f} s")
    print(f"Median delta_t: {median_t:.4f} s")
    print(f"Min delta_t: {min_t:.4f} s")
    print(f"Max delta_t: {max_t:.4f} s")
    print(f"Standard deviation of delta_t: {std_t:.4f} s")





    # Shutdown
    robot.close()

if __name__ == "__main__":
    main()
