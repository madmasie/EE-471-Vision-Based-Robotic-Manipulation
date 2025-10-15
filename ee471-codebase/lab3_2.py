# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import time
import matplotlib.pyplot as plt
import numpy
from classes.Robot import Robot
import numpy as np
import pickle

"""
This script tests the get_ik function and backtests it using the get_ee_pos function.
"""

def main():
    traj_time = 2.0                 # sec
    #poll_dt   = 0.1                 # sec

    robot = Robot()

    # Enable torque and set time-based profile

    tests = [
        np.array([274,   0, 204,  0]),
        np.array([ 16,   4, 336, 15]),
        np.array([  0, -270, 106,  0]),
    ]


    #compute joint angles for tests using get_ik() and print solutions for each test. then call get_fk() on each solution and print the resulting end-effector pose to verify it matches the original input pose
    for i, pose in enumerate(tests, start=1):
            q = robot.get_ik(pose)
            # print(q)
            print(f"Test {i}: pose={pose} -> joints={np.round(q, 3)} deg")

            fk_pose = robot.get_fk(q)
            # print(fk_pose)
            print(f"Test {i}: fk_pose={np.round(fk_pose, 3)} mm")


    # Shutdown
    robot.close()


if __name__ == "__main__":
    main()











