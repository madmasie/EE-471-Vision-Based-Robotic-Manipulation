"""
Prelab 6 - Part 1: Kabsch 3D Point Registration (A -> B)
Run this file to reproduce the results printed below.
"""
import numpy as np

def point_registration(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[0] != 3 or B.shape[0] != 3 or A.shape[1] != B.shape[1]:
        raise ValueError("A and B must be 3xn with the same n")
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    A_c = A - centroid_A
    B_c = B - centroid_B
    H = A_c @ B_c.T
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = centroid_B - R @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return T

def main():
    A = np.array([[681.2, 526.9, 914.8],
                  [542.3, 381.0, 876.5],
                  [701.2, 466.3, 951.4],
                  [598.4, 556.8, 876.9],
                  [654.3, 489.0, 910.2]]).T
    B = np.array([[110.1, 856.3, 917.8],
                  [115.1, 654.9, 879.5],
                  [167.1, 827.5, 954.4],
                  [ 30.4, 818.8, 879.9],
                  [117.9, 810.4, 913.2]]).T
    T = point_registration(A, B)
    A_h = np.vstack([A, np.ones((1, A.shape[1]))])
    A_to_B = (T @ A_h)[:3, :]
    rmse = np.sqrt(np.mean(np.sum((A_to_B - B) ** 2, axis=0)))
    R = T[:3, :3]
    RtR = R @ R.T
    detR = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(RtR - np.eye(3), ord='fro')
    np.set_printoptions(precision=4, suppress=True)
    T_out = T.copy()
    T_out[:3, :3] = np.round(T[:3, :3], 4)
    T_out[:3, 3] = np.round(T[:3, 3], 4)
    print("4x4 Transformation Matrix T (A -> B):")
    print(T_out)
    print("\nRMSE (mm): {:.4f}".format(rmse))
    print("\nOrthogonality check (||R R^T - I||_F): {:.4e}".format(orthogonality_error))
    print("det(R): {:.6f}".format(detR))

if __name__ == "__main__":
    main()
