import cv2
import dlib
from imutils import face_utils


# 右眼[37, 42], 左眼[43，48]，但由于下标是从零开始的，所以取[36, 42]和[42, 48](左闭右开)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # (42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)

def Iris_Detec(img):
    # 先设置检测器
    predictor_path = '../../shape_predictor_68_face_landmarks.dat'
    # 检测器只负责检测人脸
    detector = dlib.get_frontal_face_detector()
    # 预测器负责找到脸部的特征点
    predictor = dlib.shape_predictor(predictor_path)
    # 判断是否是灰度图
    if(len(img.shape) == 3 and img.shape[2] != 1):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 先检测到人脸
    faces = detector(img_gray, 0)
    # 首先判断是否检测到人脸
    if (len(faces) != 0):
        for index, face in enumerate(faces):
            # 面部特征点检测
            shape = predictor(img, face)
            # 把shape对象转成二维数组，这样方便找到左右眼的坐标
            shape = face_utils.shape_to_np(shape)
            # 左眼的边缘
            leftEye = shape[lStart:lEnd]
            # 求出左眼的质心
            moments_left = cv2.moments(leftEye)
            left_x = int(moments_left['m10'] / moments_left['m00'])
            left_y = int(moments_left['m01'] / moments_left['m00'])
            # 右眼的坐标
            rightEye = shape[rStart:rEnd]
            moments_right = cv2.moments(rightEye)
            right_x = int(moments_right['m10'] / moments_right['m00'])
            right_y = int(moments_right['m01'] / moments_right['m00'])
            # 左右质心的中心距
            x = (left_x + right_x) // 2
            y = (left_y + right_y) // 2

    return left_x, left_y, right_x, right_y, x, y





if __name__ == '__main__':

    # 读取图片
    img = cv2.imread("../../testPic/0001.jpg")
    # 左右眼的质心和中心
    left_x, left_y, right_x, right_y, x, y = Iris_Detec(img)
    print(left_x, left_y, right_x, right_y, x, y)




