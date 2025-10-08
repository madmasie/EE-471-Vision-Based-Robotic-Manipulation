# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Skeleton Robot class for OpenManipulator-X Robot for EE 471

import numpy as np
import math
from .OM_X_arm import OM_X_arm
from .DX_XM430_W350 import DX_XM430_W350

"""
Robot class for controlling the OpenManipulator-X Robot.
Inherits from OM_X_arm and provides methods specific to the robot's operation.
"""
class Robot(OM_X_arm):
    """
    Initialize the Robot class.
    Creates constants and connects via serial. Sets default mode and state.
    """
    def __init__(self):
        super().__init__()

        self.dim = [77, 130, 124, 126]
        theta1 = 0
        theta2 = 0
        theta3 = 0
        theta4 = 0


        # TODO: fill your DH table (4x4) based on your derivation.
        # Example DH table: [theta, d, a, alpha] for each joint (replace with your actual values)
        # self.DHTable = np.array([
        #     [theta1,         self.dim[0],      0,           -math.pi/2],
        #     [theta2 - (math.pi/2) - math.asin(24/130), 0,   self.dim[1], 0],
        #     [theta2 + (math.pi/2) - math.asin(24/130), 0,   self.dim[2], 0],
        #     [theta4,         0,                self.dim[3], 0]
        # ], dtype=float)
        # self.DHTable = np.array([
        #     [0,-((math.pi/2) - math.asin(24/130)),(math.pi/2) - math.asin(24/130),0],
        #     [self.dim[0], 0, 0, 0],
        #     [0, self.dim[1], self.dim[2], self.dim[3]],
        #     [-(math.pi/2), 0, 0, 0]
        # ], dtype=float)
        self.DHTable = np.array([[theta1, self.dim[0], 0, -np.pi/2],
                                    [theta2 - ((np.pi/2) - np.asin(24/130)), 0, self.dim[1], 0],
                                    [theta3 + ((np.pi/2) - np.asin(24/130)), 0, self.dim[2], 0],
                                    [theta4, 0, self.dim[3], 0]], dtype=float)  # replace with your table
        

        self.GRIP_OPEN_DEG  = -45.0
        self.GRIP_CLOSE_DEG = +45.0
        self.GRIP_THRESH_DEG = 180.0

        # Robot Dimensions (in mm)
        self.mDim = [77, 130, 124, 126]
        self.mOtherDim = [128, 24]
        
        # Set default mode and state
        # Change robot to position mode with torque enabled by default
        # Feel free to change this as desired
        self.write_mode('position')
        self.write_motor_state(True)

        # Set the robot to move between positions with a 5 second trajectory profile
        # change here or call writeTime in scripts to change
        self.write_time(5)

    def _set_time_profile_bit_all(self, enable: bool):
        """Turn the Drive Mode 'time-based profile' bit (bit 2) on/off for all joints."""
        DX = DX_XM430_W350
        # Read current drive modes
        dm = self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, None)  # list[int]
        if not isinstance(dm, list) or len(dm) != len(self.motorIDs):
            raise RuntimeError("Failed to read DRIVE_MODE for all joints.")
        new_dm = []
        for v in dm:
            if enable:
                new_dm.append(v | 0b100)   # set bit 2
            else:
                new_dm.append(v & ~0b100)  # clear bit 2
        # Write back (bulk)
        self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, new_dm)

    """
    Sends the joints to the desired angles.
    Parameters:
    goals (list of 1x4 float): Angles (degrees) for each of the joints to go to.
    """
    def write_joints(self, q_deg):
        """Send joint target angles in degrees (list/array length N)."""
        DX = DX_XM430_W350
        q_deg = list(q_deg)
        if len(q_deg) != len(self.motorIDs):
            raise ValueError(f"Expected {len(self.motorIDs)} joint angles, got {len(q_deg)}")

        ticks = [int(round(angle * DX.TICKS_PER_DEG + DX.TICK_POS_OFFSET)) for angle in q_deg]

        # If you're in normal position mode (not extended), keep values in [0, 4095]
        ticks = [max(0, min(int(DX.TICKS_PER_ROT - 1), t)) for t in ticks]

        self.bulk_read_write(DX.POS_LEN, DX.GOAL_POSITION, ticks)

    """
    Creates a time-based profile (trapezoidal) based on the desired times.
    This will cause write_position to take the desired number of seconds to reach the setpoint.
    Parameters:
    time (float): Total profile time in seconds. If 0, the profile will be disabled (be extra careful).
    acc_time (float, optional): Total acceleration time for ramp up and ramp down (individually, not combined). Defaults to time/3.
    """
    def write_time(self, total_time_s, acc_time_s=None):
        """Configure trapezoidal TIME profile for all joints."""
        if acc_time_s is None:
            acc_time_s = float(total_time_s) / 3.0

        # Enable time-based profile (bit 2) for all joints
        self._set_time_profile_bit_all(True)

        acc_ms = int(round(acc_time_s * DX_XM430_W350.MS_PER_S))
        tot_ms = int(round(float(total_time_s) * DX_XM430_W350.MS_PER_S))

        # Bulk write to all joints
        self.bulk_read_write(DX_XM430_W350.PROF_ACC_LEN, DX_XM430_W350.PROF_ACC, [acc_ms] * len(self.motorIDs))
        self.bulk_read_write(DX_XM430_W350.PROF_VEL_LEN, DX_XM430_W350.PROF_VEL, [tot_ms] * len(self.motorIDs))

    """
    Sets the gripper to be open or closed.
    Parameters:
    open (bool): True to set the gripper to open, False to close.
    """
    def write_gripper(self, is_open: bool):
        """Open/close gripper using fixed angles in position mode."""
        target = self.GRIP_OPEN_DEG if is_open else self.GRIP_CLOSE_DEG
        self.gripper.write_position(target)

    def read_gripper(self) -> float:
        """Return gripper joint position in degrees."""
        return self.gripper.read_position()

    def read_gripper_open(self) -> bool:
        return (self.read_gripper() > self.GRIP_THRESH_DEG)

    """
    Sets position holding for the joints on or off.
    Parameters:
    enable (bool): True to enable torque to hold the last set position for all joints, False to disable.
    """
    def write_motor_state(self, enable):
        state = 1 if enable else 0
        states = [state] * self.motorsNum  # Repeat the state for each motor
        self.bulk_read_write(DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, states)

    """
    Supplies the joints with the desired currents.
    Parameters:
    currents (list of 1x4 float): Currents (mA) for each of the joints to be supplied.
    """
    def write_currents(self, currents):
        current_in_ticks = [round(current * DX_XM430_W350.TICKS_PER_mA) for current in currents]
        self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.GOAL_CURRENT, current_in_ticks)

    """
    Change the operating mode for all joints.
    Parameters:
    mode (str): New operating mode for all joints. Options include:
        "current": Current Control Mode (writeCurrent)
        "velocity": Velocity Control Mode (writeVelocity)
        "position": Position Control Mode (writePosition)
        "ext position": Extended Position Control Mode
        "curr position": Current-based Position Control Mode
        "pwm voltage": PWM Control Mode
    """
    def write_mode(self, mode):
        if mode in ['current', 'c']:
            write_mode = DX_XM430_W350.CURR_CNTR_MD
        elif mode in ['velocity', 'v']:
            write_mode = DX_XM430_W350.VEL_CNTR_MD
        elif mode in ['position', 'p']:
            write_mode = DX_XM430_W350.POS_CNTR_MD
        elif mode in ['ext position', 'ep']:
            write_mode = DX_XM430_W350.EXT_POS_CNTR_MD
        elif mode in ['curr position', 'cp']:
            write_mode = DX_XM430_W350.CURR_POS_CNTR_MD
        elif mode in ['pwm voltage', 'pwm']:
            write_mode = DX_XM430_W350.PWM_CNTR_MD
        else:
            raise ValueError(f"writeMode input cannot be '{mode}'. See implementation in DX_XM430_W350 class.")

        self.write_motor_state(False)
        write_modes = [write_mode] * self.motorsNum  # Create a list with the mode value for each motor
        self.bulk_read_write(DX_XM430_W350.OPR_MODE_LEN, DX_XM430_W350.OPR_MODE, write_modes)
        self.write_motor_state(True)

    """
    Gets the current joint positions, velocities, and currents.
    Returns:
    numpy.ndarray: A 3x4 array containing the joints' positions (deg), velocities (deg/s), and currents (mA).
    """
    def get_joints_readings(self):
        """
        Returns a 3xN array: [deg; deg/s; mA] for the N arm joints (excludes gripper).
        """
        N = len(self.motorIDs)

        # Bulk read raw registers
        pos_u32 = self.bulk_read_write(DX_XM430_W350.POS_LEN, DX_XM430_W350.CURR_POSITION, None)  # list of ints
        vel_u32 = self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.CURR_VELOCITY, None)
        cur_u16 = self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.CURR_CURRENT,  None)

        # Vectorize
        pos_u32 = np.array(pos_u32, dtype=np.uint32)
        vel_u32 = np.array(vel_u32, dtype=np.uint32)  # signed 32-bit
        cur_u16 = np.array(cur_u16, dtype=np.uint16)  # signed 16-bit

        # Convert signed types
        vel_i32 = (vel_u32.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
        vel_i32 = vel_i32.astype(np.int32)
        cur_i16 = (cur_u16.astype(np.int32) + (1 << 15)) % (1 << 16) - (1 << 15)
        cur_i16 = cur_i16.astype(np.int16)

        # Units
        q_deg  = (pos_u32.astype(np.int64) - int(DX_XM430_W350.TICK_POS_OFFSET)) / DX_XM430_W350.TICKS_PER_DEG
        qd_dps = vel_i32 / DX_XM430_W350.TICKS_PER_ANGVEL
        I_mA   = cur_i16 / DX_XM430_W350.TICKS_PER_mA

        readings = np.vstack([q_deg.astype(float), qd_dps.astype(float), I_mA.astype(float)])
        return readings

    """
    Sends the joints to the desired velocities.
    Parameters:
    vels (list of 1x4 float): Angular velocities (deg/s) for each of the joints to go at.
    """
    def write_velocities(self, vels):
        """Send joint target velocities in deg/s (list/array length N)."""
        vels = list(vels)
        if len(vels) != len(self.motorIDs):
            raise ValueError(f"Expected {len(self.motorIDs)} velocities, got {len(vels)}")

        ticks_per_s = [int(round(v * DX_XM430_W350.TICKS_PER_ANGVEL)) for v in vels]  # signed
        self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.GOAL_VELOCITY, ticks_per_s)
    

    def get_dh_row_mat(self, row):
        """
        Standard DH homogeneous transform A_i for a single row [theta, d, a, alpha].
        All angles in radians.
        """
        # Flatten and cast to floats (prevents "array element with a sequence" errors)
        theta, d, a, alpha = map(float, np.array(row).reshape(-1)[:4])
        
        cos = math.cos
        sin = math.sin

        # Standard DH:
        # A_i =
        # [ cθ, -sθ cα,  sθ sα,  a cθ]
        # [ sθ,  cθ cα, -cθ sα,  a sθ]
        # [  0,     sα,     cα,    d  ]
        # [  0,      0,      0,    1  ]
        A = np.array([
            [cos(row[0]), -sin(row[0])*cos(row[3]), sin(row[0])*sin(row[3]), row[2]*cos(row[0])],
            [sin(row[0]), cos(row[0])*cos(row[3]), -cos(row[0])*sin(row[3]), row[2]*sin(row[0])],
            [0, sin(row[3]), cos(row[3]), row[1]],
            [0, 0, 0, 1]
        ], dtype=float)

        return A

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
        dh = np.copy(self.DHTable)
        q_rad = np.radians(joint_angles)
        dh[:, 0] += q_rad                    # add joint variables to θ
        A_stack = np.zeros((4,4,4), float)
        for i in range(4):
            A_stack[:, :, i] = self.get_dh_row_mat(dh[i, :])
        return A_stack
        # return A_stack

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
# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Skeleton Robot class for OpenManipulator-X Robot for EE 471

import numpy as np
import math
from .OM_X_arm import OM_X_arm
from .DX_XM430_W350 import DX_XM430_W350

"""
Robot class for controlling the OpenManipulator-X Robot.
Inherits from OM_X_arm and provides methods specific to the robot's operation.
"""
class Robot(OM_X_arm):
    """
    Initialize the Robot class.
    Creates constants and connects via serial. Sets default mode and state.
    """
    def __init__(self):
        super().__init__()

        self.dim = [77, 130, 124, 126]
        theta1 = 0
        theta2 = 0
        theta3 = 0
        theta4 = 0


        # TODO: fill your DH table (4x4) based on your derivation.
        # Example DH table: [theta, d, a, alpha] for each joint (replace with your actual values)
        # self.DHTable = np.array([
        #     [theta1,         self.dim[0],      0,           -math.pi/2],
        #     [theta2 - (math.pi/2) - math.asin(24/130), 0,   self.dim[1], 0],
        #     [theta2 + (math.pi/2) - math.asin(24/130), 0,   self.dim[2], 0],
        #     [theta4,         0,                self.dim[3], 0]
        # ], dtype=float)
        # self.DHTable = np.array([
        #     [0,-((math.pi/2) - math.asin(24/130)),(math.pi/2) - math.asin(24/130),0],
        #     [self.dim[0], 0, 0, 0],
        #     [0, self.dim[1], self.dim[2], self.dim[3]],
        #     [-(math.pi/2), 0, 0, 0]
        # ], dtype=float)
        self.DHTable = np.array([[theta1, self.dim[0], 0, -np.pi/2],
                                    [theta2 - ((np.pi/2) - np.asin(24/130)), 0, self.dim[1], 0],
                                    [theta3 + ((np.pi/2) - np.asin(24/130)), 0, self.dim[2], 0],
                                    [theta4, 0, self.dim[3], 0]], dtype=float)  # replace with your table
        

        self.GRIP_OPEN_DEG  = -45.0
        self.GRIP_CLOSE_DEG = +45.0
        self.GRIP_THRESH_DEG = 180.0

        # Robot Dimensions (in mm)
        self.mDim = [77, 130, 124, 126]
        self.mOtherDim = [128, 24]
        
        # Set default mode and state
        # Change robot to position mode with torque enabled by default
        # Feel free to change this as desired
        self.write_mode('position')
        self.write_motor_state(True)

        # Set the robot to move between positions with a 5 second trajectory profile
        # change here or call writeTime in scripts to change
        self.write_time(5)

    def _set_time_profile_bit_all(self, enable: bool):
        """Turn the Drive Mode 'time-based profile' bit (bit 2) on/off for all joints."""
        DX = DX_XM430_W350
        # Read current drive modes
        dm = self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, None)  # list[int]
        if not isinstance(dm, list) or len(dm) != len(self.motorIDs):
            raise RuntimeError("Failed to read DRIVE_MODE for all joints.")
        new_dm = []
        for v in dm:
            if enable:
                new_dm.append(v | 0b100)   # set bit 2
            else:
                new_dm.append(v & ~0b100)  # clear bit 2
        # Write back (bulk)
        self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, new_dm)

    """
    Sends the joints to the desired angles.
    Parameters:
    goals (list of 1x4 float): Angles (degrees) for each of the joints to go to.
    """
    def write_joints(self, q_deg):
        """Send joint target angles in degrees (list/array length N)."""
        DX = DX_XM430_W350
        q_deg = list(q_deg)
        if len(q_deg) != len(self.motorIDs):
            raise ValueError(f"Expected {len(self.motorIDs)} joint angles, got {len(q_deg)}")

        ticks = [int(round(angle * DX.TICKS_PER_DEG + DX.TICK_POS_OFFSET)) for angle in q_deg]

        # If you're in normal position mode (not extended), keep values in [0, 4095]
        ticks = [max(0, min(int(DX.TICKS_PER_ROT - 1), t)) for t in ticks]

        self.bulk_read_write(DX.POS_LEN, DX.GOAL_POSITION, ticks)

    """
    Creates a time-based profile (trapezoidal) based on the desired times.
    This will cause write_position to take the desired number of seconds to reach the setpoint.
    Parameters:
    time (float): Total profile time in seconds. If 0, the profile will be disabled (be extra careful).
    acc_time (float, optional): Total acceleration time for ramp up and ramp down (individually, not combined). Defaults to time/3.
    """
    def write_time(self, total_time_s, acc_time_s=None):
        """Configure trapezoidal TIME profile for all joints."""
        if acc_time_s is None:
            acc_time_s = float(total_time_s) / 3.0

        # Enable time-based profile (bit 2) for all joints
        self._set_time_profile_bit_all(True)

        acc_ms = int(round(acc_time_s * DX_XM430_W350.MS_PER_S))
        tot_ms = int(round(float(total_time_s) * DX_XM430_W350.MS_PER_S))

        # Bulk write to all joints
        self.bulk_read_write(DX_XM430_W350.PROF_ACC_LEN, DX_XM430_W350.PROF_ACC, [acc_ms] * len(self.motorIDs))
        self.bulk_read_write(DX_XM430_W350.PROF_VEL_LEN, DX_XM430_W350.PROF_VEL, [tot_ms] * len(self.motorIDs))

    """
    Sets the gripper to be open or closed.
    Parameters:
    open (bool): True to set the gripper to open, False to close.
    """
    def write_gripper(self, is_open: bool):
        """Open/close gripper using fixed angles in position mode."""
        target = self.GRIP_OPEN_DEG if is_open else self.GRIP_CLOSE_DEG
        self.gripper.write_position(target)

    def read_gripper(self) -> float:
        """Return gripper joint position in degrees."""
        return self.gripper.read_position()

    def read_gripper_open(self) -> bool:
        return (self.read_gripper() > self.GRIP_THRESH_DEG)

    """
    Sets position holding for the joints on or off.
    Parameters:
    enable (bool): True to enable torque to hold the last set position for all joints, False to disable.
    """
    def write_motor_state(self, enable):
        state = 1 if enable else 0
        states = [state] * self.motorsNum  # Repeat the state for each motor
        self.bulk_read_write(DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, states)

    """
    Supplies the joints with the desired currents.
    Parameters:
    currents (list of 1x4 float): Currents (mA) for each of the joints to be supplied.
    """
    def write_currents(self, currents):
        current_in_ticks = [round(current * DX_XM430_W350.TICKS_PER_mA) for current in currents]
        self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.GOAL_CURRENT, current_in_ticks)

    """
    Change the operating mode for all joints.
    Parameters:
    mode (str): New operating mode for all joints. Options include:
        "current": Current Control Mode (writeCurrent)
        "velocity": Velocity Control Mode (writeVelocity)
        "position": Position Control Mode (writePosition)
        "ext position": Extended Position Control Mode
        "curr position": Current-based Position Control Mode
        "pwm voltage": PWM Control Mode
    """
    def write_mode(self, mode):
        if mode in ['current', 'c']:
            write_mode = DX_XM430_W350.CURR_CNTR_MD
        elif mode in ['velocity', 'v']:
            write_mode = DX_XM430_W350.VEL_CNTR_MD
        elif mode in ['position', 'p']:
            write_mode = DX_XM430_W350.POS_CNTR_MD
        elif mode in ['ext position', 'ep']:
            write_mode = DX_XM430_W350.EXT_POS_CNTR_MD
        elif mode in ['curr position', 'cp']:
            write_mode = DX_XM430_W350.CURR_POS_CNTR_MD
        elif mode in ['pwm voltage', 'pwm']:
            write_mode = DX_XM430_W350.PWM_CNTR_MD
        else:
            raise ValueError(f"writeMode input cannot be '{mode}'. See implementation in DX_XM430_W350 class.")

        self.write_motor_state(False)
        write_modes = [write_mode] * self.motorsNum  # Create a list with the mode value for each motor
        self.bulk_read_write(DX_XM430_W350.OPR_MODE_LEN, DX_XM430_W350.OPR_MODE, write_modes)
        self.write_motor_state(True)

    """
    Gets the current joint positions, velocities, and currents.
    Returns:
    numpy.ndarray: A 3x4 array containing the joints' positions (deg), velocities (deg/s), and currents (mA).
    """
    def get_joints_readings(self):
        """
        Returns a 3xN array: [deg; deg/s; mA] for the N arm joints (excludes gripper).
        """
        N = len(self.motorIDs)

        # Bulk read raw registers
        pos_u32 = self.bulk_read_write(DX_XM430_W350.POS_LEN, DX_XM430_W350.CURR_POSITION, None)  # list of ints
        vel_u32 = self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.CURR_VELOCITY, None)
        cur_u16 = self.bulk_read_write(DX_XM430_W350.CURR_LEN, DX_XM430_W350.CURR_CURRENT,  None)

        # Vectorize
        pos_u32 = np.array(pos_u32, dtype=np.uint32)
        vel_u32 = np.array(vel_u32, dtype=np.uint32)  # signed 32-bit
        cur_u16 = np.array(cur_u16, dtype=np.uint16)  # signed 16-bit

        # Convert signed types
        vel_i32 = (vel_u32.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
        vel_i32 = vel_i32.astype(np.int32)
        cur_i16 = (cur_u16.astype(np.int32) + (1 << 15)) % (1 << 16) - (1 << 15)
        cur_i16 = cur_i16.astype(np.int16)

        # Units
        q_deg  = (pos_u32.astype(np.int64) - int(DX_XM430_W350.TICK_POS_OFFSET)) / DX_XM430_W350.TICKS_PER_DEG
        qd_dps = vel_i32 / DX_XM430_W350.TICKS_PER_ANGVEL
        I_mA   = cur_i16 / DX_XM430_W350.TICKS_PER_mA

        readings = np.vstack([q_deg.astype(float), qd_dps.astype(float), I_mA.astype(float)])
        return readings

    """
    Sends the joints to the desired velocities.
    Parameters:
    vels (list of 1x4 float): Angular velocities (deg/s) for each of the joints to go at.
    """
    def write_velocities(self, vels):
        """Send joint target velocities in deg/s (list/array length N)."""
        vels = list(vels)
        if len(vels) != len(self.motorIDs):
            raise ValueError(f"Expected {len(self.motorIDs)} velocities, got {len(vels)}")

        ticks_per_s = [int(round(v * DX_XM430_W350.TICKS_PER_ANGVEL)) for v in vels]  # signed
        self.bulk_read_write(DX_XM430_W350.VEL_LEN, DX_XM430_W350.GOAL_VELOCITY, ticks_per_s)
    

    def get_dh_row_mat(self, row):
        """
        Standard DH homogeneous transform A_i for a single row [theta, d, a, alpha].
        All angles in radians.
        """
        # Flatten and cast to floats (prevents "array element with a sequence" errors)
        theta, d, a, alpha = map(float, np.array(row).reshape(-1)[:4])
        
        cos = math.cos
        sin = math.sin

        # Standard DH:
        # A_i =
        # [ cθ, -sθ cα,  sθ sα,  a cθ]
        # [ sθ,  cθ cα, -cθ sα,  a sθ]
        # [  0,     sα,     cα,    d  ]
        # [  0,      0,      0,    1  ]
        A = np.array([
            [cos(row[0]), -sin(row[0])*cos(row[3]), sin(row[0])*sin(row[3]), row[2]*cos(row[0])],
            [sin(row[0]), cos(row[0])*cos(row[3]), -cos(row[0])*sin(row[3]), row[2]*sin(row[0])],
            [0, sin(row[3]), cos(row[3]), row[1]],
            [0, 0, 0, 1]
        ], dtype=float)

        return A

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
        dh = np.copy(self.DHTable)
        q_rad = np.radians(joint_angles)
        dh[:, 0] += q_rad                    # add joint variables to θ
        A_stack = np.zeros((4,4,4), float)
        for i in range(4):
            A_stack[:, :, i] = self.get_dh_row_mat(dh[i, :])
        return A_stack
        # return A_stack

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
    
    def get_current_fk(self):
        return self.get_fk(self.get_joints_readings()[0, :])  # first row is q (deg)

    def get_ee_pos(self):
        q_deg = self.get_joints_readings()[0, :]  # first row is q (deg)
        T = self.get_fk(q_deg)
        x, y, z = T[0:3, 3]
        yaw = q_deg[0]
        pitch = -(q_deg[1] + q_deg[2] + q_deg[3])
        return np.array([x, y, z, pitch, yaw], dtype=float)

