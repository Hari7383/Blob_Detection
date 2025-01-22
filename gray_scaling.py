import cv2

cap = cv2.VideoCapture(1)
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('frame', thresh)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    gray1 = gray2

cap.release()
cv2.destroyAllWindows()