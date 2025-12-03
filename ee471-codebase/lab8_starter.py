"""
Lab 8 (Final Project): Vision-Guided Robotic Pick-and-Place Sorting System
Team: [Your Team Name]
Members: [Team Member Names]

This script implements a complete robotic sorting system that:
1. Detects colored balls using computer vision
2. Localizes them in 3D space using camera-robot calibration
3. Plans smooth trajectories to pick up each ball
4. Sorts them into color-coded bins

System Architecture:
    Detection → Localization → Motion Planning → Execution → Repeat
"""
import numpy as np
import cv2
import time
import pickle

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.TrajPlanner import TrajPlanner



#prelab 8 stuff
FONT = cv2.FONT_HERSHEY_SIMPLEX

def morph_open_close(mask: np.ndarray, k_open: int = 5, k_close: int = 7) -> np.ndarray:
    kernel_open = np.ones((k_open, k_open), np.uint8)
    kernel_close = np.ones((k_close, k_close), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return cleaned

def circularity(area: float, perimeter: float) -> float:
    """
    4 * pi * Area / Perimeter^2 : close to 1.0 for circles.
    """
    if perimeter <= 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters
BALL_RADIUS = 15  # Physical radius of balls in millimeters

# Motion control parameters
TRAJECTORY_TIME = 2.5  # Time for each trajectory segment in seconds
NUM_POINTS = 100       # Number of waypoints in each trajectory

# Workspace safety bounds (millimeters, in robot frame)
# TODO: Adjust these based on your setup to prevent collisions
X_MIN, X_MAX = 30, 230   # Forward/backward limits
Y_MIN, Y_MAX = -150, 250 # Left/right limits

# Home position: [x, y, z, pitch] in mm and degrees
# This position should give the camera a clear view of the workspace
HOME_POSITION = [100, 0, 220, -15]

# Sorting bin locations: [x, y, z, pitch] in mm and degrees
# TODO: Adjust these positions based on your physical bin locations
BINS = {
    'red': [0, -220, 150, -40],
    'orange': [120, -220, 150, -40],
    'blue': [0, 220, 150, -45],
    'yellow': [120, 220, 150, -45]
}

# ============================================================================
# COMPUTER VISION: BALL DETECTION AND POSE ESTIMATION
# ============================================================================

def get_ball_pose(corners: np.ndarray, intrinsics: any, radius: float) -> tuple:
    """
    Estimate the 3D pose of a detected sphere using the Perspective-n-Point (PnP) algorithm.
    
    The PnP algorithm finds the position and orientation of an object by matching:
    - Known 3D points on the object (object_points)
    - Corresponding 2D points in the image (image_points)
    
    For a sphere, we create artificial "corner" points on the sphere's visible boundary
    to establish these correspondences.
    
    Args:
        corners: A 4x2 array of circle boundary points [top, bottom, left, right]
        intrinsics: Camera intrinsic parameters from RealSense
        radius: The sphere's physical radius in millimeters
    
    Returns:
        tuple: (rotation_matrix, translation_vector)
            - rotation_matrix: 3x3 matrix (not used for spheres, but returned by PnP)
            - translation_vector: 3x1 vector giving sphere center in camera frame (mm)
    
    Raises:
        RuntimeError: If PnP algorithm fails to find a solution
    """
    # ==========================================================================
    # TODO: Define 3D object points on the sphere
    # ==========================================================================
    # Hint: Create 4 points on the sphere's "equator" as viewed from camera
    # These should correspond to the top, bottom, left, and right boundary points
    # Example structure:
    #   Point 1: [-radius, 0, 0]  (left edge of sphere)
    #   Point 2: [+radius, 0, 0]  (right edge of sphere)
    #   Point 3: [0, +radius, 0]  (top edge of sphere)
    #   Point 4: [0, -radius, 0]  (bottom edge of sphere)
    # YOUR CODE HERE
    object_points = np.array([
        [-radius, 0, 0],   # Left
        [ radius, 0, 0],   # Right
        [0,  radius, 0],   # Top
        [0, -radius, 0]    # Bottom
    ], dtype=np.float32).reshape(-1, 1, 3)
    
    # ==========================================================================
    # TODO: Prepare image points (the detected corner points in pixels)
    # ==========================================================================
    # Hint: Reshape corners array to (4, 2) and ensure float32 type
    # YOUR CODE HERE 
    top, bottom, left, right = corners
    image_points = np.array([
        left,    # Left
        right,   # Right
        top,     # Top
        bottom   # Bottom
    ], dtype=np.float32).reshape(-1, 1, 2)   
    

    # ==========================================================================
    # TODO: Construct camera intrinsic matrix
    # ==========================================================================
    # The camera matrix format is:
    #   [[fx,  0, cx],
    #    [ 0, fy, cy],
    #    [ 0,  0,  1]]
    # where fx, fy are focal lengths and cx, cy are principal point coordinates
    # Hint: Access intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    # YOUR CODE HERE
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
    
    
    # ==========================================================================
    # TODO: Solve PnP to get rotation and translation
    # ==========================================================================
    # Hint: Use cv2.solvePnP()
    # Returns: (success, rvec, tvec) where rvec is rotation vector, tvec is translation
    # YOUR CODE HERE
    
    # Convert rotation vector to rotation matrix (required return format)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("PnP solution could not be found")
    
    rot_matrix, _ = cv2.Rodrigues(rvec)

    return rot_matrix, tvec

def detect_balls(image):
    """
    Detect colored balls in the input image using computer vision.
    
    Pipeline:
        1. Convert image to HSV color space for robust color detection
        2. Detect circular shapes using Hough Circle Transform
        3. For each detected circle:
           - Extract the color by analyzing hue values
           - Classify as red, orange, yellow, or blue
           - Record position and radius
    
    Args:
        image: BGR color image from camera
    
    Returns:
        list: List of tuples (color, (cx, cy), radius) for each detected ball
              Returns None if no balls detected
    """
    # ==========================================================================
    # TODO: DETECT CIRCULAR OBJECTS IN IMAGE
    # ==========================================================================
    # Implement your circle detection pipeline here. Consider approaches such as:
    #   - Hough Circle Transform (cv2.HoughCircles)
    #   - Contour detection (cv2.findContours) with circularity filtering
    #   - Color-based segmentation followed by shape analysis
    #   - Edge detection + morphological operations
    #
    # Your implementation should:
    #   - Detect circles reliably under varying lighting conditions
    #   - Handle multiple balls of different colors
    #   - Return circle parameters (center x, y and radius) for each detection
    #
    # Recommended: Store results in a variable called 'circles' with format:
    #              array of shape (1, N, 3) where each circle is [x, y, radius]
    #              (This matches cv2.HoughCircles output format)
    #
    # Explore different preprocessing techniques and parameters!
    
    # Preprocessing for better detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    
    # Color ranges from prelab8 (better than simple hue thresholding)
    COLOR_RANGES = {
        "red": [
            (np.array([0,   120, 70]),  np.array([4, 255, 255])),
            (np.array([170, 120, 70]),  np.array([180, 255, 255])),
        ],
        "orange": [(np.array([5, 120, 70]), np.array([22, 255, 255]))],
        "yellow": [(np.array([23, 120, 70]), np.array([35, 255, 255]))],
        "blue":   [(np.array([90, 120, 60]), np.array([125, 255, 255]))],
    }
    
    result = []
    min_area = 150
    min_circ = 0.70
    
    # Try contour-based detection first (more robust)
    for cname, ranges in COLOR_RANGES.items():
        # Create color mask
        mask = np.zeros(hsv_blur.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_blur, lo, hi))
        
        # Clean up with morphology
        mask = morph_open_close(mask, 5, 7)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_this_color = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            circ = circularity(area, perimeter)
            if circ < min_circ:
                continue
                
            # Get center and radius
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = int(np.sqrt(area / np.pi))
            
            result.append((cname, (cx, cy), radius))
            found_this_color = True
        
        # Fallback to Hough circles if contours didn't work
        if not found_this_color:
            masked_gray = cv2.bitwise_and(gray_blur, gray_blur, mask=mask)
            circles = cv2.HoughCircles(
                masked_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=40,
                param1=120,
                param2=28,
                minRadius=12,
                maxRadius=80
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    result.append((cname, (circle[0], circle[1]), circle[2]))
    
    # If no color-based detection worked, try general Hough circles
    if not result:
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=120,
            param2=28,
            minRadius=12,
            maxRadius=80
        )
        
        if circles is None or len(circles[0]) == 0:
            cv2.imshow('Detection', image)
            cv2.waitKey(1)
            return None
        
        circles = np.uint16(np.around(circles))
        
        # Process each detected circle
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # ======================================================================
            # TODO: CLASSIFY BALL COLOR
            # ======================================================================
            # Determine the color of this detected circle. Approaches to consider:
            #   - HSV color space analysis (hue values)
            #
            # Important: Focus analysis on pixels within the circle (using masks), not entire image
            #
            # Must classify as one of: 'red', 'orange', 'yellow', 'blue'
            # Set variable 'color' to the classification result
            # Use 'continue' to skip circles that don't match any target color
            #
            # Note: Red color may require special handling depending on color space!
            #
            # YOUR CODE HERE

            # Create circular mask for this circle
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            # Check each color range for best match
            color = None
            best_overlap = 0
            
            for cname, ranges in COLOR_RANGES.items():
                color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for (lo, hi) in ranges:
                    range_mask = cv2.inRange(hsv, lo, hi)
                    color_mask = cv2.bitwise_or(color_mask, range_mask)
                
                # Check overlap between circle and color mask
                overlap = cv2.bitwise_and(mask, color_mask)
                overlap_ratio = np.sum(overlap) / np.sum(mask)
                
                if overlap_ratio > best_overlap and overlap_ratio > 0.3:  # 30% threshold
                    best_overlap = overlap_ratio
                    color = cname

            # Skip if no color matched well enough
            if color is None:
                continue
            
            # Add to results
            result.append((color, center, int(radius)))
            
            # ======================================================================
            # TODO: Draw detection visualization on image
            # ======================================================================
            # Hint: Draw circle outline, center point, and color label
            # Use cv2.circle() for shapes and cv2.putText() for text
            # YOUR CODE HERE
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Green circle
            cv2.circle(image, center, 3, (0, 0, 255), -1)      # Red center dot
            cv2.putText(image, color, (center[0] - 20, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # Draw visualization for contour-based detections
        for color, center, radius in result:
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Green circle
            cv2.circle(image, center, 3, (0, 0, 255), -1)      # Red center dot
            cv2.putText(image, color, (center[0] - 20, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display detection results
    cv2.imshow('Detection', image)
    cv2.waitKey(1)
    
    return result if result else None


# ============================================================================
# MOTION CONTROL: TRAJECTORY PLANNING AND EXECUTION
# ============================================================================

def move_trajectory(robot, target_pos, traj_time=TRAJECTORY_TIME):
    """
    Move robot to target position using smooth quintic trajectory.
    
    This function:
    1. Gets current robot position
    2. Plans a quintic (5th-order polynomial) trajectory in task space
    3. Converts entire trajectory to joint space using inverse kinematics
    4. Executes trajectory with precise timing
    
    Args:
        robot: Robot instance
        target_pos: Target position [x, y, z, pitch] in mm and degrees
        traj_time: Duration of trajectory in seconds
    """


    max_samples = 10000
    count = 0  # Sample counter
    data_time = np.zeros(max_samples)
    data_pid = np.zeros((max_samples, 3))           # PID output velocity commands
    data_joint_vel = np.zeros((max_samples, 4))     # Joint velocities 
    pos_current = np.zeros((max_samples, 4))        # Current EE positions
    pos_desired = np.zeros((max_samples, 4))        # Desired EE positions
    data_error = np.zeros((max_samples, 3))      # Position errors
    tag_detected_list = [] 
    # ==========================================================================
    # TODO: Get current joint positions and end-effector position
    # ==========================================================================
    q_curr = robot.get_joints_readings()[0, :]  # Current joint angles
    current_pos = robot.get_ee_pos(q_curr)[0:4]     # Current end-effector pose [x, y, z, pitch]
    print(current_pos)
    print(target_pos)
    #target_pos = np.array(target_pos, dtype=np.float32)
    
    
    # ==========================================================================
    # TODO: Create task-space trajectory from current to target position
    # ==========================================================================
    # Hint: Stack current_pos and target_pos, create TrajPlanner, call get_quintic_traj()
    # YOUR CODE HERE
    waypoints = np.vstack((current_pos, target_pos))
    planner = TrajPlanner(waypoints)
    trajectories = planner.get_quintic_traj(traj_time, NUM_POINTS)  #shape NUM_POINTS,4
    
    
    # ==========================================================================
    # TODO: Convert entire trajectory to joint space
    # ==========================================================================
    joint_trajectories = []
    for i in range(len(trajectories)):
        t = trajectories[i, 0]
        x, y, z, phi = trajectories[i, 1:]
        q = robot.get_ik([x, y, z, phi])
        joint_trajectories.append([t, q[0], q[1], q[2], q[3]])
    trajectories = np.array(joint_trajectories)
    # joint_traj = []
    # for p in traj_task:
    #     q = robot.get_ik(p.tolist())       # IK for each task-space point
    #     joint_traj.append(q)
    # joint_traj = np.array(joint_traj)
    
    # TODO: Calculate time step between waypoints
    dt = traj_time / (NUM_POINTS - 1)
    robot.write_time(dt)
    
    # ==========================================================================
    # TODO: Execute trajectory with precise timing (see Lab 4)
    # ==========================================================================
    # Hint: Record start time, then for each waypoint:
    #   - Calculate target time for this waypoint
    #   - Wait until that time
    #   - Send joint commands
    # This ensures smooth, consistent motion regardless of computation time
    # YOUR CODE HERE

    start_time = time.perf_counter()

    for i in range(1, len(trajectories)):
        # Calculate when this command should be sent
        target_time = start_time + trajectories[i, 0]
        
        # Wait until it's time to send this command
        while time.perf_counter() < target_time:
            # Collect data while waiting
             current_time = time.perf_counter() - start_time
            
            # if count < max_samples:
            #     data_q[count, :] = robot.get_joints_readings()[0, :]
            #     data_time[count] = current_time
            #     data_ee_poses[count, :] = robot.get_ee_pos(data_q[count, :])[0:4]
            #     count += 1
            # # Small sleep to preventfget CPU overload
            # time.sleep(0.001)  # 1ms sleep
        
        # Send the command at the scheduled time
        robot.write_joints(trajectories[i, 1:])

        if count < max_samples:
                    data_time[count] = time.perf_counter() - start_time
                    
                    
                    pos_current[count, :] = current_pos
                    pos_desired[count, :] = target_pos

    data = {
            'time': data_time, # Timestamp (s)
            'pos_current': pos_current, # Current EE position [x,y,z] (mm)
            'pos_desired': pos_desired, # Desired EE position [x,y,z] (mm)
            'error': data_error, # Position error [ex,ey,ez] (mm)
            'control_output': data_pid, # PID output [vx,vy,vz] (mm/s)
            'joint_vel': data_joint_vel, # Joint velocities (rad/s)
            'tag_detected': tag_detected_list  # Boolean: tag visible
        }
            
    filename = 'lab8_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved! Collected {count} samples")


    # start_time = time.time()
    # for i, q in enumerate(traj):
    #     target_time = start_time + i * dt
    #     while time.time() < target_time:
    #         time.sleep(0.001)  # Sleep briefly to avoid busy-waiting
    #     robot.write_joint_(q.tolist())


# ============================================================================
# PICK AND PLACE OPERATIONS
# ============================================================================

def pick_ball(robot, ball_pos):
    """
    Execute a pick operation to grasp a ball.
    
    Sequence:
        1. Open gripper
        2. Move to approach position (above ball)
        3. Move down to grasp position
        4. Close gripper
        5. Lift ball to clear workspace
    
    Args:
        robot: Robot instance
        ball_pos: Ball position [x, y, z] in robot frame (mm)
    """
    print(f"Picking ball at {ball_pos}")
    
    # ==========================================================================
    # Open gripper
    # ==========================================================================
    # Use robot.write_gripper(1) for open, wait for motion
    robot.write_gripper(1)
    time.sleep(0.5)
    
    # ==========================================================================
    # Move to approach position (above the ball)
    # ==========================================================================
    # Create position [x, y, z_high, pitch] where z_high is ~100mm
    # Use steep pitch angle (e.g., -80°) for vertical approach
    approach = [ball_pos[0], ball_pos[1], 100, -80]   # may need tuning
    move_trajectory(robot, approach, TRAJECTORY_TIME)
    print("Reached approach position")
    # ==========================================================================
    # Move down to grasp position
    # ==========================================================================
    # Lower z to just above table surface (e.g., 39mm for 30mm ball radius)
    # Adjust this height based on your table height and ball size!
    grasp = [ball_pos[0], ball_pos[1], 39, -80]       # may need tuning
    move_trajectory(robot, grasp, TRAJECTORY_TIME * 0.5)
    print("Reached grasp position")
    
    # ==========================================================================
    # Close gripper to grasp ball
    # ==========================================================================
    #  Use robot.write_gripper(0) for close, wait for secure grasp
    robot.write_gripper(0)
    time.sleep(1)
    
    # ==========================================================================
    # Lift ball to clear workspace
    # ==========================================================================
    # Move back up to approach height
    lift = [ball_pos[0], ball_pos[1], 100, -80] # may need tuning
    move_trajectory(robot, lift, TRAJECTORY_TIME * 0.5)


def place_ball(robot, color):
    """
    Place ball in the appropriate color-coded bin.
    
    Args:
        robot: Robot instance
        color: Ball color string ('red', 'orange', 'yellow', 'blue')
    """
    print(f"Placing {color} ball")
    
    # ==========================================================================
    # Get bin position for this color
    # ==========================================================================
    # Look up position in BINS dictionary
    bin_pos = BINS[color]
    
    # ==========================================================================
    # Move to bin location
    # ==========================================================================
    move_trajectory(robot, bin_pos, TRAJECTORY_TIME)
    
    # ==========================================================================
    # Release ball by opening gripper
    # ==========================================================================
    robot.write_gripper(1)
    time.sleep(1)


def go_home(robot):
    """
    Return robot to home position for next detection cycle.
    
    Args:
        robot: Robot instance
    """
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME)


# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def main():
    """
    Main control loop for the robotic sorting system.
    
    Workflow:
        1. Initialize robot, camera, and calibration
        2. Move to home position
        3. Loop:
           a. Capture image and detect balls
           b. Convert detected positions to robot frame
           c. Filter balls within workspace
           d. Pick and place first ball
           e. Repeat
    """
    print("="*60)
    print("Lab 8: Robotic Sorting System")
    print("="*60)
    
    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
                         # Tag detection status
    # TODO: Initialize robot, camera, and get intrinsics
    # Hint: Create Robot(), Realsense(), and get intrinsics
    # YOUR CODE HERE
    robot = Robot()
    camera = Realsense()
    intrinsics = camera.get_intrinsics()
    
    
    # ==========================================================================
    # TODO: Load camera-robot calibration matrix
    # ==========================================================================
    # Hint: Use np.load() to load 'camera_robot_transform.npy'
    # This matrix transforms points from camera frame to robot frame
    # YOUR CODE HERE
    T_cam_to_robot = np.load('camera_robot_transform.npy')
    
    
    # ==========================================================================
    # TODO: Setup robot in position control mode
    # ==========================================================================
    # Hint: Set mode to "position", enable motors, set default trajectory time
    # YOUR CODE HERE
    robot.write_mode("position")
    robot.write_motor_state(True)
    robot.write_time(5.0)
    
    # ==========================================================================
    # TODO: Move to home position
    # ==========================================================================
    # Hint: Use inverse kinematics to find joint angles, then command them
    # YOUR CODE HERE
    q_home = robot.get_ik(HOME_POSITION)
    robot.write_joints(q_home)
    time.sleep(5)  # Wait for motion to complete
    
    # ==========================================================================
    # TODO: Open gripper initially
    # ==========================================================================
    # YOUR CODE HERE
    robot.write_gripper(1)
    time.sleep(0.5)
    
    print(f"\nReady! Using TRAJECTORY control")
    print("Press Ctrl+C to stop\n")
    
    # ==========================================================================
    # MAIN CONTROL LOOP
    # ==========================================================================
    
    try:
        iteration = 0
        
        while True:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            
            # ==================================================================
            # STEP 1: CAPTURE IMAGE AND DETECT BALLS
            # ==================================================================
            
            # TODO: Get camera frame
            # Hint: Use camera.get_frames() which returns (color, depth)
            # YOUR CODE HERE
            color_frame, depth_frame = camera.get_frames()
            image = color_frame
            
            
            # TODO: Detect balls in image
            # Hint: Call detect_balls() function
            # YOUR CODE HERE
            spheres = detect_balls(image)
            
            # Check if any balls detected
            if spheres is None:
                print("No balls detected")
                time.sleep(1)
                iteration += 1
                continue
            
            print(f"Detected {len(spheres)} ball(s)")
            
            # ==================================================================
            # STEP 2: CONVERT DETECTIONS TO ROBOT FRAME
            # ==================================================================
            
            robot_spheres = []  # List to store (color, robot_position) tuples
            
            for color, (cx, cy), radius in spheres:
                
                # ==============================================================
                # TODO: Create corner points for PnP algorithm
                # ==============================================================
                # Hint: Create 4 points at [left, right, bottom, top] of circle
                # Format: [[cx - radius, cy], [cx + radius, cy], ...]
                # YOUR CODE HERE
                corners = np.array([
                    [cx,         cy - radius],  # Top
                    [cx,         cy + radius],  # Bottom
                    [cx - radius, cy        ],  # Left
                    [cx + radius, cy        ]   # Right
                ], dtype=np.float32)
                
                
                # ==============================================================
                # TODO: Get 3D position in camera frame using PnP
                # ==============================================================
                # Hint: Call get_ball_pose() with corners, intrinsics, and BALL_RADIUS
                # Returns (rotation, translation) - we only need translation
                # YOUR CODE HERE     
                rot_cam, t_cam = get_ball_pose(corners, intrinsics, BALL_RADIUS)
                cam_pos = t_cam.flatten()  # 3D position in camera frame (mm)           
                
                # ==============================================================
                # TODO: Transform position to robot frame
                # ==============================================================
                # Hint: 
                #   1. Flatten cam_pos and append 1 for homogeneous coordinates
                #   2. Multiply by transformation matrix: T_cam_to_robot @ pos_hom
                #   3. Extract first 3 elements for 3D position
                # YOUR CODE HERE
                cam_pos_hom= np.hstack((cam_pos, 1))  # make homogeneous
                robot_pos_hom = T_cam_to_robot @ cam_pos_hom
                robot_pos = robot_pos_hom[:3]
                
                
                # ==============================================================
                # Check if position is within workspace bounds
                # ==============================================================
                # Check if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX
                # Skip balls outside workspace for safety
                if not (X_MIN <= robot_pos[0] <= X_MAX and 
                        Y_MIN <= robot_pos[1] <= Y_MAX):
                    print(f"  Skipping {color} ball outside workspace: {robot_pos}")
                    continue
                
                
                robot_spheres.append((color, robot_pos))
                print(f"  {color}: {robot_pos}")
            
            # Check if any valid balls found
            if not robot_spheres:
                print("No balls in workspace")
                time.sleep(1)
                iteration += 1
                continue
            
            # ==================================================================
            # STEP 3: PICK AND PLACE FIRST BALL
            # ==================================================================
            
            # Get first ball from list
            color, pos = robot_spheres[0]
            
            print(f"\nSorting {color} ball at {pos}")
            
            # TODO: Execute pick-and-place sequence
            # Hint: Call pick_ball(), place_ball(), and go_home()
            # YOUR CODE HERE
            pick_ball(robot, pos)
            place_ball(robot, color)    
            go_home(robot)  
            
            iteration += 1
            time.sleep(1)  # Brief pause before next cycle
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        camera.stop()
        
        cv2.destroyAllWindows()
        print("Done!")


        
        
        # Then do the cleanup
        print("\nStopping robot and cleaning up...")
        plot_data('lab8_data.pkl')
        robot.write_velocities([0, 0, 0, 0])
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")


def plot_data(filename='lab8_data.pkl'):
    """
    Load data and create required plots per lab requirements.
    Trims unused zero rows from preallocated arrays so plots look clean.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Load data
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return

    # Extract raw arrays
    time_data = np.array(data['time'])
    pos_current = np.array(data['pos_current'])
    pos_desired = np.array(data['pos_desired'])
    error_data = np.array(data['error'])
    control_output = np.array(data['control_output'])
    joint_vel = np.array(data['joint_vel'])
    tag_detected = np.array(data['tag_detected'], dtype=bool)  # list -> array

    # -------------------------------------------------------------------------
    # Trim to only valid samples (non-zero positions OR positive time)
    # -------------------------------------------------------------------------
    # A row is "valid" if either pos_current or pos_desired is nonzero
    pos_nonzero = np.any(pos_current != 0, axis=1) | np.any(pos_desired != 0, axis=1)
    # Also accept any row with time > 0 (in case your very first sample has zeros)
    time_nonzero = time_data > 0

    valid_mask = pos_nonzero | time_nonzero
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("plot_data: No valid samples found (all zeros).")
        return

    start = valid_indices[0]
    end = valid_indices[-1] + 1  # slice end is exclusive

    # Slice everything consistently
    time_data = time_data[start:end]
    pos_current = pos_current[start:end, :]
    pos_desired = pos_desired[start:end, :]
    error_data = error_data[start:end, :]
    control_output = control_output[start:end, :]
    joint_vel = joint_vel[start:end, :]

    # If tag_detected is shorter (it's a list with length=count), align it
    if tag_detected.size >= end:
        tag_detected = tag_detected[start:end]
    else:
        # Just truncate/match to time length
        tag_detected = tag_detected[: time_data.shape[0]]

    # -------------------------------------------------------------------------
    # 5. 3D TRAJECTORY - single figure
    # -------------------------------------------------------------------------
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')

    # Filter again in case some leading points are still all-zero
    valid_current = pos_current[np.any(pos_current != 0, axis=1)]
    valid_desired = pos_desired[np.any(pos_desired != 0, axis=1)]

    if len(valid_current) > 1:
        ax5.plot(valid_current[:, 0], valid_current[:, 1], valid_current[:, 2],
                 'b-', linewidth=3, label='Actual Path', alpha=0.8)

        ax5.scatter(valid_current[0, 0], valid_current[0, 1], valid_current[0, 2],
                    color='green', s=150, label='Start', marker='o', edgecolor='black')
        ax5.scatter(valid_current[-1, 0], valid_current[-1, 1], valid_current[-1, 2],
                    color='red', s=150, label='End', marker='s', edgecolor='black')

    if len(valid_desired) > 1:
        ax5.plot(valid_desired[:, 0], valid_desired[:, 1], valid_desired[:, 2],
                 'r--', linewidth=2, label='Desired Path', alpha=0.7)

    if len(valid_current) > 0:
        all_points = valid_current
        if len(valid_desired) > 0:
            all_points = np.vstack([valid_current, valid_desired])

        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax5.set_xlim(mid_x - max_range, mid_x + max_range)
        ax5.set_ylim(mid_y - max_range, mid_y + max_range)
        ax5.set_zlim(mid_z - max_range, mid_z + max_range)

    ax5.set_xlabel('X Position (mm)')
    ax5.set_ylabel('Y Position (mm)')
    ax5.set_zlabel('Z Position (mm)')
    ax5.set_title('3D End-Effector Trajectory')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()