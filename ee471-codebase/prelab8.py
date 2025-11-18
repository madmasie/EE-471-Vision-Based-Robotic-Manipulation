# EE 470 / 471 — Prelab 8: Color Ball Detection
#funcs from lecctures:
#   - Color conversion to HSV, cv2.inRange, cv2.bitwise_or  (Lec19)
#   - Gaussian blur & morphologyEx Open/Close              (Lec20)
#   - findContours, arcLength, contourArea, moments        (Lec21)
#   - HoughCircles fallback with slide-style args (Lec21)

import cv2
import numpy as np
from typing import List, Tuple

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

def detect_balls(image_path: str,
                 draw: bool = True) -> List[Tuple[str, Tuple[int, int], int]]:
    """
      1) Read image: GaussianBlur (Lec21)
      2) BGR→HSV: color masks with cv2.inRange (Lec19)
      3) Morphological OPEN/CLOSE (Lec20)
      4) cv2.findContours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) (Lec21)
      5) Filter by area & circularity; centroid from cv2.moments (Lec21)
      6) Fallback: cv2.HoughCircles with slide-style params (Lec21)

    Returns: [(color, (x,y), r), ...] where (x,y) is centroid in pixels.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    vis = img.copy()

    # 1) Smooth: prepare grayscale for Hough, HSV for color masking
    gray_for_hough = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             
    gray_for_hough = cv2.GaussianBlur(gray_for_hough, (5, 5), 1.2)     
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                          
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)                               

    # 2) Color masks using inRange; red uses two ranges (HSV wrap) 
    COLOR_RANGES = {
        "red": [
            (np.array([0,   120, 70]),  np.array([10, 255, 255])),
            (np.array([170, 120, 70]),  np.array([180, 255, 255])),
        ],
        "orange": [(np.array([11, 120, 70]), np.array([22, 255, 255]))],
        "yellow": [(np.array([23, 120, 70]), np.array([35, 255, 255]))],
        "blue":   [(np.array([90, 120, 60]), np.array([125,255, 255]))],
    }

    #detections will hold tuples of (color_name, (cx, cy), radius)
    detections: List[Tuple[str, Tuple[int, int], int]] = []
    min_area = 150            # quick noise gate (scale-dependent)
    min_circ = 0.70           #near 1.0 for circles

    # Process each color
    for cname, ranges in COLOR_RANGES.items():
        # Combine masks if the color has multiple hue bands (red)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi)) 

        # 3) Morphological OPEN/CLOSE cleanup
        mask = morph_open_close(mask, 5, 7)

        # 4) Find contours 
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

        found_this_color = False
        for c in cnts:
            # 5) Area, Perimeter, Circularity 
            area = cv2.contourArea(c)                                   
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)                               
            circ = circularity(area, peri)                              

            if circ < min_circ:
                continue

            # Moments -> centroid 
            M = cv2.moments(c)                                          
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])                               
            cy = int(M["m01"] / M["m00"])                               

            r  = int(np.sqrt(area / np.pi))                           

            detections.append((cname, (cx, cy), r))
            found_this_color = True

            if draw:
                cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)            
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)          
                cv2.putText(vis, cname, (cx + 8, cy - 8), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, cname, (cx + 8, cy - 8), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 6) Hough fallback if no contour passed our filters
        if not found_this_color:
            # Restrict Hough to this masked region
            masked = cv2.bitwise_and(gray_for_hough, gray_for_hough, mask=mask)
            circles = cv2.HoughCircles(                                  
                masked, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=40,
                param1=120, param2=28,
                minRadius=12, maxRadius=80
            )
            # If circles found, add them
            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                for (x, y, r) in circles:
                    detections.append((cname, (int(x), int(y)), int(r)))
                    if draw:
                        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
                        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
                        cv2.putText(vis, cname, (int(x)+8, int(y)-8), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(vis, cname, (int(x)+8, int(y)-8), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Output annotated image
    if draw:
        cv2.imwrite("prelab8_output.png", vis)
        cv2.imshow("Annotated", vis); print("Saved prelab8_output.png"); cv2.waitKey(0); cv2.destroyAllWindows()

    return detections

if __name__ == "__main__":
    image_path = "ee471-codebase/image_prelab8.jpg"

    results = detect_balls(image_path, draw=True)

    for color, (x, y), _ in sorted(results, key=lambda t: t[0]):
        print(f"{color.capitalize()}: ({x}, {y})")
