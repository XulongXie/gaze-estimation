import cv2
import os


output_dir = "C:/Users/hasee/Desktop/Master_Project/Step1/DataSet"
i = 1
cap = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror
    first_point = (480, 10)
    last_point = (1440, 1000)
    cv2.rectangle(frame, first_point, last_point, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    flag = cv2.waitKey(1)
    if flag == 13:
        """Press Enter to take pictures"""
        output_path = os.path.join(output_dir, "%04d.jpg" % i)
        cv2.imwrite(output_path, frame)
        i += 1
    if flag == 27:
        """Press ESC to exit"""
        break
