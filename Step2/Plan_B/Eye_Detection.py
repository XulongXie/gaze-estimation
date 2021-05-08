import numpy as np
import cv2
import dlib
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

# 右眼[37, 42], 左眼[43，48]，但由于下标是从零开始的，所以取[38, 42]和[42, 48](左闭右开)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # (42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)

def Eye_Detec(img):
    # 先设置检测器
    predictor_path = '../../shape_predictor_68_face_landmarks.dat'
    # 检测器只负责检测人脸
    detector = dlib.get_frontal_face_detector()
    # 预测器负责找到脸部的特征点
    predictor = dlib.shape_predictor(predictor_path)
    # 防止内存泄露，先深拷贝一下
    img_show = img.copy()
    # 判断是否是灰度图
    if(len(img.shape) == 3 and img.shape[2] != 1):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 用一个list去存放脸的坐标点，这样以后方便使用
    pos_list = []
    # 先检测到人脸
    faces = detector(img_gray, 0)
    # 首先判断是否检测到人脸
    if (len(faces) != 0):
        for index, face in enumerate(faces):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, face).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                # 存起来
                pos_list.append(pos)
            # 面部特征点检测
            shape = predictor(img, face)
            # 把shape对象转成二维数组，这样方便找到左右眼的坐标
            shape = face_utils.shape_to_np(shape)
            # 左右眼的坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # 左眼的眼角
            left_left = leftEye[0, 0]
            left_right = leftEye[3, 0]
            left_top = max(leftEye[1, 1], leftEye[2, 1])
            left_down = max(leftEye[4, 1], leftEye[5, 1])
            left_upper_left = (left_left - 5, left_top - 5)
            left_down_right = (left_right + 5, left_down + 5)
            left = [left_upper_left, left_down_right]
            # 左眼眼角画框
            cv2.rectangle(img_show, left_upper_left, left_down_right, (0, 0, 255), 2)
            # 右眼的眼角
            right_left = rightEye[0, 0]
            right_right = rightEye[3, 0]
            right_top = max(rightEye[1, 1], rightEye[2, 1])
            right_down = max(rightEye[4, 1], rightEye[5, 1])
            right_upper_left = (right_left - 5, right_top - 5)
            right_down_right = (right_right + 5, right_down + 5)
            right = [right_upper_left, right_down_right]
            # 右眼眼角画框
            cv2.rectangle(img_show, right_upper_left, right_down_right, (0, 0, 255), 2)
            # 检测有几张人脸
            cv2.putText(img_show, 'There are {} faces in the picture.'.format(len(faces)),
                        (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # 单独提取人眼部分, 裁剪坐标为[y0:y1, x0:x1]
            left_eye = img_show[left[0][1]:left[1][1], left[0][0]:left[1][0]]
            right_eye = img_show[right[0][1]:right[1][1], right[0][0]:right[1][0]]

    return left[0][0], left[0][1], left[1][0], left[1][1], right[0][0], right[0][1], right[1][0], right[1][1]


if __name__ == '__main__':
    # 读取图片
    img = cv2.imread("../../testPic/000449.jpg")
    # 人眼检测
    x0, y0, x1, y1, x2, y2, x3, y3 = Eye_Detec(img)
    EyeCorners = np.array([x0, y0, x1, y1, x2, y2, x3, y3])
    # 显示
    print(EyeCorners)
    print(type(EyeCorners))
''' 
    cv2.namedWindow("left", 0)
    cv2.resizeWindow("left", 64, 64)
    cv2.imshow('left', left_eye)
    cv2.namedWindow("right", 0)
    cv2.resizeWindow("right", 64, 64)
    cv2.imshow('right', right_eye)
    key = cv2.waitKey(0)
'''


