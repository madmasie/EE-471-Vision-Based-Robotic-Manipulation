import numpy as np
from classes.PIDController import PIDController
import matplotlib.pyplot as plt

''' Create a PIDController instance (see the Test Setup below)
• Simulate a simple motion scenario with known errors (see the Test Loop below)
• Apply your PID computation
• Plot the system response over time
Hints:
• You may need the following NumPy operations:
– Matrix multiplication with @ operator
– Element-wise multiplication with *
• Remember to update error_prev and error_integral appropriately
• The control period dt is used for both integral and derivative calculations'''

#test setup
# Initialize controller with 50ms control period
controller = PIDController(dim=3, dt=0.05)

# Time vector (5 seconds of simulation)
t = np.linspace(0, 5, 801)

# Initialize arrays to store results
errors = np.zeros((len(t), 3))
outputs = np.zeros((len(t), 3))

# Set initial error (different for each axis to test independently)
error = np.array([10., 8., 6.]) # mm

#test loop
# Simulate system response
for i in range(len(t)):
    # Store current error
    errors[i] = error

    # Compute PID control output (desired velocity in mm/s)
    outputs[i] = controller.compute_pid(error)

    # Simple model: error reduces due to control action
    # Error reduction = velocity * timestep
    error = error - outputs[i] * 0.05

# Plot the results here ...
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, errors)
plt.title('Position Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (mm)')
plt.legend(['X Error', 'Y Error', 'Z Error'])
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t, outputs)
plt.title('PID Control Output (Velocity Command) Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm/s)')
plt.legend(['X Velocity', 'Y Velocity', 'Z Velocity'])
plt.grid()
plt.tight_layout()
plt.show()


