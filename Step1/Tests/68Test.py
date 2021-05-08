import dlib
import numpy as np
import cv2
import time
from imutils import face_utils
from scipy.spatial import distance as dist

# 睁眼率：每只眼睛由6个（x，y）坐标表示，从眼睛的左角开始，然后围绕该区域的其余部分顺时针显示
# ||P2 -P6|| + ||P3 - P5|| / 2 * ||P4 - P1||
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 摄像头的默认分辨率，要知道这两个值才方便写东西
WINDOWS_WIDTH = 640
WINDOWS_HEIGHT = 480
# 当EAR低于某个阈值时，眼睛处于闭合状态
EYE_AR_THRESH = 0.2
# 设置在3帧或3帧以上时，认为眼睛处于闭合，这样可以检测眨眼或有效区分眨眼还是闭眼
EYE_AR_CONSEC_FRAMES = 2
# 帧数计数器
COUNTER = 0
# 闭眼或眨眼次数
TOTAL = 0
FPS = 0

predictor_path = '../../shape_predictor_68_face_landmarks.dat'
# 检测器只负责检测人脸
detector = dlib.get_frontal_face_detector()
# 预测器负责找到脸部的特征点
predictor = dlib.shape_predictor(predictor_path)

# 右眼[37, 42], 左眼[43，48]，但由于下标是从零开始的，所以取[38, 42]和[42, 48](左闭右开)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # (42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)

cap = cv2.VideoCapture(0)
while (True):
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 返回类似bonding box的矩形框
    faces = detector(img, 1)
    # 首先判断是否检测到人脸
    if(len(faces) != 0):
        # 对人脸一个一个进行处理
        for index, face in enumerate(faces):
            # 找到bonding box的左下右上的坐标点，也可以用左上右下
            right_top = (face.right(), face.top())
            left_bottom = (face.left(), face.bottom())
            # 画面部检测框
            cv2.rectangle(frame, right_top, left_bottom, (255, 0, 0), 2)

            # 面部特征点检测
            shape = predictor(img, face)
            # 这里做的相当于一个数据结构的转换，把shape对象里面的点对象按顺序依次读取出来并放进一个矩阵中
            # 这里用landmarks = face_utils.shape_to_np(shape)也是可以的，但是后面的pos要变成(point[0], point[1])
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            # 把这68个点做画圈处理
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                cv2.circle(frame, pos, 2, (0, 255, 0), -1)
            # 把shape对象转成二维数组，这样方便找到左右眼的坐标
            shape = face_utils.shape_to_np(shape)
            # 左右眼的坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # 计算睁眼率
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 平均睁眼率
            ear = (leftEAR + rightEAR) / 2.0
            # 找出眼部的凸包，返回的是一个3维的ndarray数据格式的一系列点，其实也没有必要，因为68点中眼部只有外围点
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓，最后一个1设置为非填充模式
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # 检测眨眼
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
            # 0.5是font_size, 1是font_thickness, cv2.LINE_AA是线条种类
            cv2.putText(frame, 'EAR: {:.2f}'.format(ear), (WINDOWS_WIDTH - 120, 20 + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # 这里根据每一帧判断是睁眼还是闭眼
            cv2.putText(frame, 'State: {}'.format('OPEN' if ear > 0.2 else 'CLOSE'),
                        (WINDOWS_WIDTH - 120, 20 + 30 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)
            # 统计眨眼的次数，其实眨眼的算法还不算好
            cv2.putText(frame, 'Blind: {}'.format(TOTAL), (20, 20 + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        end_time = time.time()
        # 每秒可以检测的帧数
        FPS = 1 / (end_time - start_time)
        # 检测有几张人脸
        cv2.putText(frame, 'There are {} faces in the picture.'.format(len(faces)),
                    (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # 显示FPS
        cv2.putText(frame, 'FPS: {:.4f}'.format(FPS), (WINDOWS_WIDTH - 120, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.namedWindow("Cap", 0)
        cv2.resizeWindow("Cap", WINDOWS_WIDTH, WINDOWS_HEIGHT)
        cv2.imshow('Cap', frame)
        key = cv2.waitKey(5)
        # 按下Esc键退出
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
