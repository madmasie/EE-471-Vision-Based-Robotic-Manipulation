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

def main():
    traj_time = 2.0                 # sec
    poll_dt   = 0.5                 # sec

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    #send the arm to its zero pose.
    np.set_printoptions(precision=3, suppress=True)

    # print(robot.get_fk([0, 0, 0, 0]))
    # print(robot.get_fk([15, -45, -60, 90]))
    # print(robot.get_fk([-90, 15, 30, -45]))
    print(robot.get_current_fk())
    print(robot.get_ee_pos())
    robot.write_joints([0, 0, 0, 0])
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < traj_time:
        print(robot.get_current_fk())
        print(robot.get_ee_pos())
        time.sleep(poll_dt)
    time.sleep(traj_time)
    
    # Shutdown
    robot.close()

if __name__ == "__main__":
    main()