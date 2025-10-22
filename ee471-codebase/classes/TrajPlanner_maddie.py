import numpy as np

class TrajPlanner:
    """
    Trajectory Planner class for calculating trajectories for different polynomial orders and relevant coefficients.
    """

    def __init__(self, setpoints):
        """
        Initialize the TrajPlanner class.

        Parameters:
        setpoints (numpy array): List of setpoints to travel to.
        """
        self.setpoints = setpoints

    ## Implement the required methods below. ##
    def calc_cubic_coeff(self, t0, tf, p0, pf, v0, vf):
            """
            Given the initial time, final time, initial position, final position, initial velocity, and final velocity,
            returns cubic polynomial coefficients.

            Parameters:
            t0 (float): Start time of trajectory
            tf (float): End time of trajectory
            p0 (float): Initial setpoint
            pf (float): Final setpoint
            v0 (float): Initial velocity
            vf (float): Final velocity

            Returns:
            1x4 numpy array: The calculated polynomial coefficients for the desired cubic trajectory.
            """

            T = float(tf - t0)
            if T <= 0:
                raise ValueError("tf must be > t0")

            # Build system at tau = 0 and tau = T
            A = np.array([
                [1, 0,     0,      0     ],   # p(0)  = p0
                [1, T,   T**2,   T**3   ],   # p(T)  = pf
                [0, 1,     0,      0     ],   # p'(0) = v0
                [0, 1,   2*T,   3*T**2  ]    # p'(T) = vf
            ], dtype=float)

            b = np.array([p0, pf, v0, vf], dtype=float)

            a = np.linalg.solve(A, b) #a = inverse(A) @ b
            return a  # [a0, a1, a2, a3] for p(tau)
            

    def calc_cubic_traj(self, traj_time, points_num, coeff):
            """
            Given the time between setpoints, number of points between waypoints, and polynomial coefficients,
            returns the cubic trajectory of waypoints for a single pair of setpoints.

            Parameters:
            traj_time (int): Time between setPoints.
            points_num (int): Number of waypoints between setpoints.
            coeff (numpy array): Polynomial coefficients for trajectory.

            Returns:
            numpy array:  (n + 2) x 1 array of waypoints for the cubic trajectory.
            """

            # coeff = [a0, a1, a2, a3]
            a0 = coeff[0]
            a1 = coeff[1]
            a2 = coeff[2]
            a3 = coeff[3]

            t0 = 0
            t_values = np.linspace(t0, traj_time, points_num + 2)

            #empty list of waypoints to store
            waypoints = []

            #for each time value, calculate position using cubic polynomial
            for t in t_values:
                pos = a0 + a1*t + a2*(t**2) + a3*(t**3)

                # append position to waypoints list
                waypoints.append(pos)

            # turn list into column vector
            waypoints = np.array(waypoints)

            # reshape to (n+2) * 1
            waypoints = waypoints.reshape(len(waypoints), 1)

            return waypoints



    def get_cubic_traj(self, traj_time, points_num):
            """
            Given the time between setpoints and number of points between waypoints, returns the cubic trajectory.

            Parameters:
            traj_time (int): Time between setPoints.
            points_num (int): Number of waypoints between setpoints.

            Returns:
            numpy array: (n + 2) x 5 np array List of waypoints for the cubic trajectory specifying planned waypoints for each joint including time
            """
            # make sure there are exactly 2 setpoints (start and end)
            if self.setpoints is None:
                raise ValueError("setpoints not set yet")

            if len(self.setpoints) != 2:
                raise ValueError("setpoints must have 2 rows")

            # define time info
            t0 = 0
            tf = traj_time
            v0 = 0
            vf = 0

            # create time column
            #t_values shape: (points_num + 2) x 1
            t_values = np.linspace(t0, tf, points_num + 2)
            t_values = t_values.reshape(len(t_values), 1)

            # split start and end points
            p_start = self.setpoints[0] #index 0 is start point
            p_end   = self.setpoints[1] #index 1 is end point

            # list to hold all columns [time, q1, q2, q3, q4]
            all_cols = [t_values]

            # loop through each of the 4 joints
            for i in range(4):
                p0 = p_start[i]
                pf = p_end[i]

                #call calc_cubic_coeff to get coefficients
                coeff = self.calc_cubic_coeff(t0, tf, p0, pf, v0, vf)

                #call calc_cubic_traj to get trajectory for this joint
                q_values = self.calc_cubic_traj(traj_time, points_num, coeff)

                #append joint trajectory to all_cols
                all_cols.append(q_values)

            # combine all columns horizontally
            traj = np.hstack(all_cols)

            return traj



            


# # Usage Example
# import numpy as np
# from traj_planner import TrajPlanner

# # Define setpoints (example values)
# setpoints = np.array([
#     [15, -45, -60, 90],
#     [-90, 15, 30, -45]
# ])

# # Create a TrajPlanner object
# trajectories = TrajPlanner(setpoints)

# # Generate cubic trajectory
# cubic_traj = trajectories.get_cubic_traj(traj_time=5, points_num=10)
# print(cubic_traj)
