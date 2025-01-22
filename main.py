import cv2
import numpy as np

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

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints = detector.detect(gray_frame)

   
    small_blobs = [kp for kp in keypoints if kp.size < 20]  #change the Threshold value for small blobs
    big_blobs = [kp for kp in keypoints if kp.size >= 20]   

    print(f"Small Blobs: {len(small_blobs)}, Big Blobs: {len(big_blobs)}")

    output_frame = frame.copy()
    output_frame = cv2.drawKeypoints(
        #big blobs
        output_frame, small_blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    output_frame = cv2.drawKeypoints(
        #small blobs
        output_frame, big_blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("Blob Classification", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()