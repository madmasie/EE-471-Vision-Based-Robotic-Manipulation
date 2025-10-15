# # ik_test.py
# import numpy as np
# from ik import get_ik

# tests = [
#     np.array([274,   0, 204,  0]),   # home pose [0, 0, 0, 0]
#     np.array([ 16,   4, 336, 15]),   # [15, -45, -60, 90]
#     np.array([  0, -270, 106,  0]),  # [-90, 15, 30, -45]
# ]

# #for each test pose, compute and print only one solution
# for i, pose in enumerate(tests, start=1):

#     #try both solutions, but only print one
#     try:
#         q = get_ik(pose)
#         print(f"Test {i}: pose={pose} -> joints={np.round(q, 3)} deg")
#     except ValueError as e:
#         print(f"Test {i}: ERROR {e}")
