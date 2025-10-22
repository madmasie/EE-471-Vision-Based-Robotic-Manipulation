import numpy as np
from classes.TrajPlanner import TrajPlanner

#This file demonstrates functionaliy of the TrajPlanner class, specifically the get_cubic_traj method.

#create object of TrajPlanner class with 2 sets of given setpoints
def main(): 
    setpoints = np.array([
        [15, -45, -60, 90],
        [-90, 15, 30, -45]
    ], dtype=float)


    traj_time = 5.0
    points_num = 6

    planner = TrajPlanner(setpoints)


#call get_cubic_traj() method with appropriate params to generate cubic rajectories between the two setpoints, which will create 4 trajectories (one for each joint)
    cubic_traj = planner.get_cubic_traj(traj_time, points_num)



#includes print statement inside calc_cubic_coeff() method that outputs the polynomial coefficients for each joint's cubic trajectory to the terminal

    print ("\nTrajectory (time, q1, q2, q3, q4):")
    print (np.array_str(cubic_traj, precision=3, suppress_small=True))

if __name__ == "__main__":
    main()
