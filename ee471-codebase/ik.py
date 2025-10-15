# import math
# import numpy as np

# #link lengths
# DIM = (77, 130, 124, 126, 128, 24)  # (L1, L2, L3, L4, l21, l22)

# def get_ik(pose, dim=DIM):
#     """
#     Input:  pose = [x, y, z, lam_deg]  (mm, deg)
#     Output: tuple of (solution1, solution2)
#              where solution1 = elbow-up, solution2 = elbow-down
#     Raises: ValueError if unreachable
#     """
#     L1, L2, L3, L4, l21, l22 = dim
#     x, y, z, alpha_deg = pose 

#     joint_limits = [(-90, 90), (-9120, 90), (-90, 75), (-100, 100)]

#     alpha = math.radians(alpha_deg)

#     # q1
#     q1 = math.degrees(math.atan2(y, x))

#     # wrist center
#     r  = math.hypot(x, y)
#     rw = r - L4 * math.cos(alpha)
#     zw = z - L1 - L4 * math.sin(alpha)
#     dw = math.hypot(rw, zw)
#     mew = math.atan2(zw, rw)

#     # triangle relations (law of cosines)
#     cos_alpha = (L2**2 + L3**2 - dw**2) / (2.0 * L2 * L3)     # elbow interior
#     cos_gamma = (dw**2 + L2**2 - L3**2) / (2.0 * dw * L2)     # shoulder triangle

#     # numerical safety
#     cos_alpha = max(-1.0, min(1.0, cos_alpha))
#     cos_gamma = max(-1.0, min(1.0, cos_gamma))

#     if not (-1.0 <= cos_alpha <= 1.0) or not (-1.0 <= cos_gamma <= 1.0):
#         raise ValueError("Pose unreachable for given link lengths.")

#     alpha = math.acos(cos_alpha)
#     gamma = math.acos(cos_gamma)

#     # forearm offset
#     delta = math.atan2(l22, l21)

#     # elbow-up branch
#     q2_up = math.degrees(math.pi/2 - delta - gamma - mew)
#     q3_up = math.degrees(math.pi/2 + delta - alpha)
#     q4_up = -alpha_deg - q2_up - q3_up

#     # elbow-down branch
#     q2_dn = math.degrees(math.pi/2 - delta + gamma - mew)
#     q3_dn = math.degrees(math.pi/2 + delta + alpha)
#     q4_dn = -alpha_deg - q2_dn - q3_dn

#     sol1 = np.array([q1, q2_up, q3_up, q4_up], dtype=float)
#     sol2 = np.array([q1, q2_dn, q3_dn, q4_dn], dtype=float)

#     return sol1, sol2

# #rounding stuff
# np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: f"{x:.3f}"})


# #testing the three poses
# if __name__ == "__main__":
#     tests = [
#         np.array([274,   0, 204,  0]),
#         np.array([ 16,   4, 336, 15]),
#         np.array([  0, -270, 106,  0]),
#     ]

#     #for each test pose, compute and print both solutions
#     for i, pose in enumerate(tests, start=1):
#         try:
#             sol1, sol2 = get_ik(pose)
#             print(f"\nTest {i}: pose={pose}")
#             print("  Solution 1 (elbow-up):   ", np.round(sol1, 3), "deg")
#             print("  Solution 2 (elbow-down): ", np.round(sol2, 3), "deg")
#         except ValueError as e:
#             print(f"Test {i}: ERROR {e}")
