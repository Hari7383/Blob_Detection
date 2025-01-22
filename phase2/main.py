import cv2
import numpy as np

# Blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50
params.maxArea = 2000

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.87

params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

# Open the camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    blob = (255 - blob)

    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Classify blobs based on contour area
    small_contours = []
    big_contours = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Small blob threshold
            small_contours.append(cnt)
        else:
            big_contours.append(cnt)

    print(f"Small Blobs (Contours): {len(small_contours)}, Big Blobs (Contours): {len(big_contours)}")

    # Draw contours
    result = frame.copy()
    cv2.drawContours(result, small_contours, -1, (0, 255, 0), 2)  # Green for small blobs
    cv2.drawContours(result, big_contours, -1, (0, 0, 255), 2)  # Red for big blobs

    # Detect blobs using SimpleBlobDetector
    keypoints = detector.detect(blob)
    small_blobs = [kp for kp in keypoints if kp.size < 20]
    big_blobs = [kp for kp in keypoints if kp.size >= 20]

    print(f"Small Blobs (Keypoints): {len(small_blobs)}, Big Blobs (Keypoints): {len(big_blobs)}")

    # Draw blobs on the original frame
    output_frame = frame.copy()
    output_frame = cv2.drawKeypoints(
        output_frame, small_blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    output_frame = cv2.drawKeypoints(
        output_frame, big_blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display images
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Contours Result", result)
    cv2.imshow("Blob Classification", output_frame)

    # Exit on 'e' key
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
