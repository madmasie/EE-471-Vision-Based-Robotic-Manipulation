import numpy as np
import math
from math import *

class Robot:
    def __init__(self):
        """
        Initialize robot constants and the DH table.
        """
        # TODO: set any link lengths you need (in mm), if helpful
        L1_mm = 77.0
        L2_mm = 130
        L3_mm = 124
        L4_mm = 126
        theta1 = 0
        theta2 = 0
        theta3 = 0
        thata4 = 0
        self.dim = [L1_mm, L2_mm, L3_mm, L4_mm]

        # TODO: fill your DH table (4x4) based on your derivation.
        # Create the DH parameter table
        self.DHTable = np.array([
            [theta1, L1_mm, 0, -pi/2],
            [theta2-(pi/2 - asin(24/130)), 0, L2_mm, 0],
            [theta3-(pi/2 - asin(24/130)), 0, L3_mm, 0],
            [thata4, 0, L4_mm, 0]
        ], dtype=float)


    def get_dh_row_mat(self, row):
        """
        Compute the Standard DH homogeneous transform A_i from a single DH row.                                             

        Parameters
        ----------
        row : array-like, shape (4,)
            [theta, d, a, alpha] for one joint.

        Returns
        -------
        A : ndarray, shape (4,4)
        """

        # TODO: implement A
        # A = np.array([...], dtype=float)
        # return A
        A = np.array([
            [cos(row[0]), -sin(row[0])*cos(row[3]), sin(row[0])*sin(row[3]), row[2]*cos(row[0])],
            [sin(row[0]), cos(row[0])*cos(row[3]), -cos(row[0])*sin(row[3]), row[2]*sin(row[0])],
            [0, sin(row[3]), cos(row[3]), row[1]],
            [0, 0, 0, 1]
        ], dtype=float)


    def get_int_mat(self, joint_angles):
        """
        Build all intermediate DH transforms A_i for the provided joint angles.
        Parameters
        ----------
        joint_angles : array-like, shape (4,)
            Joint variables q in degrees: [q1, q2, q3, q4].

        Returns
        -------
        A_stack : ndarray, shape (4,4,4)
            A_stack[:, :, i] = A_{i+1}

        Steps
        -----
        1) Copy the base DH table.
        2) Add q (deg) to the theta column (col 0).
        3) For each row, compute A_i via get_dh_row_mat(...).
        """

        # TODO: add q to theta column (degrees)

        #copied base DH table
        dh = np.copy(self.DHTable)

        q_deg = np.radians(joint_angles)  #converting joint angles to numpy array
        #adding q to theta column (col 0)
        dh[:, 0] += q_deg

        #for each row, compute A_i via get_dh_row_mat(...)
        # TODO: build A_stack
        A_stack = np.zeros((4, 4, 4), dtype=float)  
        for i in range(4):
            A_stack[:, :, i] = self.get_dh_row_mat(dh[i, :])
        return A_stack # return A_stack

    def get_fk(self, joint_angles):
        """
        Forward kinematics to the end-effector.

        Parameters
        ----------
        joint_angles : array-like, shape (4,)
            Joint variables q in degrees.

        Returns
        -------
        T : ndarray, shape (4,4)
            Homogeneous transform T^0_4 (base to end-effector).
        """
        # TODO:
        A_stack = self.get_int_mat(joint_angles)
        T = np.eye(4, dtype=float)
        for i in range(4):
             T = T @ A_stack[:, :, i]
        return T
