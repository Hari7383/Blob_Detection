import cv2
import numpy as np

cap = cv2.VideoCapture(1)
while True:
    ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    blob = (255 - blob)

    cnts = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    big_contour = max(cnts, key=cv2.contourArea)

    blob_area_thresh = 1000
    blob_area = cv2.contourArea(big_contour)
    if blob_area < blob_area_thresh:
        print("Blob Is Too Small")

    result = frame1.copy()
    cv2.drawContours(result, [big_contour], -1, (0,0,255), 1)

    cv2.imwrite("doco3_threshold.jpg", thresh)
    cv2.imwrite("doco3_blob.jpg", blob)
    cv2.imwrite("doco3_contour.jpg", result)

    cv2.imshow("IMAGE", frame1)
    cv2.imshow("THRESHOLD", thresh)
    cv2.imshow("BLOB", blob)
    cv2.imshow("RESULT", result)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()