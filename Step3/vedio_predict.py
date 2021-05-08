import colorsys
import os
import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from Step2.models.model import Network
from Step3.visualization import visual, drawRect

from Step2.preProcessing.PreProcess import GrayImg, lightRemove, gamma_trans, letter_box


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class gazeNet(object):
    _defaults = {
        # 这里还需要修改
        "model_path": 'model_data/yolo4_weights.pth',
        "classes_path": 'model_data/myGaze.txt',
        # 输入图片的尺寸和通道数
        "model_image_size": (224, 224, 1),
        # 有CUDA的话需要设为true
        "cuda": False,
        # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，是否有用还有待测试
        "letterbox_image": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化网络
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # 获得所有的分类
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    # 生成模型
    def generate(self):
        # 建立网络
        self.network = Network().eval()

        # 载入网络的权重
        print('Loading weights into state dict...')
        # 留下接口方便以后用GPU训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location = device)
        self.network.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.network = nn.DataParallel(self.network)
            self.network = self.network.cuda()

        # 画框设置不同的颜色，每个类使用一种画框
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # 检测图片
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        # 对输入图片进行一系列的预处理
        # 首先做灰度化
        if(len(image_shape) == 3):
            img = GrayImg(image)
            # 计算均值
            mean = np.mean(img)
            # 计算gamma值
            gamma_val = math.log10(0.5) / math.log10(mean / 255)
            # 将三通道图片做gamma变换
            image_gamma = gamma_trans(image, gamma_val)
            # 变换后再转成灰度图
            image_gamma_correct = GrayImg(image_gamma)
        else:
            img = image
            # 计算均值
            mean = np.mean(img)
            # 计算gamma值
            gamma_val = math.log10(0.5) / math.log10(mean / 255)
            # 将单通道图片做gamma变换
            image_gamma_correct = gamma_trans(img, gamma_val)
        # 消除光照的影响
        img_gamma_Remove = lightRemove(image_gamma_correct)
        # 给图像增加灰条，实现不失真的resize，也可以直接resize进行识别
        if self.letterbox_image:
            crop_img = np.array(letter_box(img_gamma_Remove, [self.model_image_size[0], self.model_image_size[0]]))
        else:
            crop_img = img_gamma_Remove.resize((self.model_image_size[0], self.model_image_size[1]), Image.BICUBIC)
        # 归一化
        photo = np.array(crop_img, dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))

        # 添加上batch_size和channel维度
        photo = photo.reshape((1, photo.shape[0], photo.shape[1]))
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()
            # 将图像输入网络当中进行预测！
            outputs = self.network(images)
            pred = outputs.argmax(dim = 1).item()

        return pred


if __name__ == '__main__':
    gaze_net = gazeNet()
    capture = cv2.VideoCapture(0)
    # fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 镜像
        frame = cv2.flip(frame, 1)
        first_point = (128, 80)
        last_point = (512, 460)
        # 剪切
        crop = frame[first_point[1]:last_point[1], first_point[0]:last_point[0]]
        # 格式转变，BGRtoRGB
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # 转变成Image
        img_new = Image.fromarray(np.uint8(img))
        # 进行检测
        pred = np.array(gaze_net.detect_image(img_new))
        # 计算坐标
        cord = visual(pred)
        # 可视化
        drawRect(cord, pred, gaze_net.colors)

        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %.2f" % (fps))

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break



