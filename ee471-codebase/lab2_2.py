      
# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import time
import matplotlib.pyplot as plt
import numpy
from classes.Robot import Robot
import numpy as np
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
    np.set_printoptions(precision=3, suppress=True)
    print("Homing to [0, 0, 0, 0] deg ...")
    print(robot.get_fk([0, 0, 0, 0]))
    print(robot.get_fk([15, -45, -60, 90]))
    print(robot.get_fk([-90, 15, 30, -45]))
    time.sleep(traj_time)
    
    # Shutdown
    robot.close()

if __name__ == "__main__":
    main()