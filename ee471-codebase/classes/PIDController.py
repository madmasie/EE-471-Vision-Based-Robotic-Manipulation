import numpy as np


class PIDController:
    def __init__(self, dim=3, dt=0.05):
        # Initialize gains (tuned for position control in mm)
        self.Kp = 0.5 * np.eye(dim)  # Proportional gain
        self.Ki = 0.05 * np.eye(dim)  # Integral gain
        self.Kd = 0.1 * np.eye(dim)  # Derivative gain

        # Initialize error terms
        self.error_integral = np.zeros(dim)
        self.error_prev = np.zeros(dim)
        self.dt = dt  # Control period in seconds

    def compute_pid(self, error):
        '''Input: A 3 × 1 numpy array error, where:
        * error: Current position error in x, y, z coordinates
        * Output: 3 × 1 numpy array representing the computed velocity command
        * Description: Calculates the PID control output using the current error and error history
        * The method should compute the integral term by accumulating errors over time
        * Calculate the derivative term using the difference between current and previous er-
        ror, Combine all terms using the pre-defined gain matrices Kp, Ki, and Kd
        '''
        # Proportional term
        P = self.Kp @ error

        # Integral term
        self.error_integral += error * self.dt
        I = self.Ki @ self.error_integral

        # Derivative term
        D = self.Kd @ ((error - self.error_prev) / self.dt)
        self.error_prev = error

        # PID output
        output = P + I + D

        return output
