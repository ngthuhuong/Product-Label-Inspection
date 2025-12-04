import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera index {i} OPENED")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Cam {i}", frame)
            cv2.waitKey(1000)  # hiện 1s để nhìn
        cap.release()
    else:
        print(f"Camera index {i} FAILED")

cv2.destroyAllWindows()
