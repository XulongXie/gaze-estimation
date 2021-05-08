import cv2
import os

output_dir = "C:/Users/hasee/Desktop/Master_Project/Step1/DataSet"
i = 1
cap = cv2.VideoCapture(0)
print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror
    first_point = (128, 80)
    last_point = (512, 460)
    cv2.rectangle(frame, first_point, last_point, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    flag = cv2.waitKey(1)
    if flag == 13:
        """Press Enter to take picture"""
        crop = frame[80:460, 128:512]
        output_path = os.path.join(output_dir, "%04d.jpg" % i)
        cv2.imwrite(output_path, crop)
        i += 1
    if flag == 27:
        """Press exit"""
        break

