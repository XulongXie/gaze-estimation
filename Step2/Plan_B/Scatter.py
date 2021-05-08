import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from Step2.Plan_B.detect_Iris import Iris_Detec as detector

def XY_Cord(array):
    X = []
    Y = []
    # array中返回给我们的是(列, 行)坐标点
    for item in array:
        x = item[0]
        y = item[1]
        X.append(x)
        Y.append(y)
    return X, Y


def plot(X, Y):
    # 设置x轴和y轴的步长和起始，终点位置，还需要做修改
    # plt.figure(figsize=(10, 15), dpi=100)  # figure做画布，可要可不要
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax = 224, xmin = 0)
    plt.ylim(ymax = 224, ymin = 0)
    '''
    plt.xticks(range(0, 20, 1))
    plt.yticks(range(0, 20, 1))
    '''
    # 点的颜色，还要做修改，可以查询：https://blog.csdn.net/w576233728/article/details/86538060
    colors1 = '#FF0000'
    colors2 = '#FFA500'
    colors3 = '#008000'
    colors4 = '#0000FF'
    colors5 = '#A0522D'
    colors6 = '#800080'
    colors7 = '#000000'
    colors8 = '#FFFF00'
    colors9 = '#FFC0CB'
    # 点面积，可能也需要更改
    area = np.pi * 1 ** 2
    # 画散点图，其中s是点的大小，alpha是点的透明度
    plt.scatter(X[0], Y[0], s = area, c = colors1, alpha = 0.4, label = 'Class A')
    plt.scatter(X[1], Y[1], s = area, c = colors2, alpha = 0.4, label = 'Class B')
    plt.scatter(X[2], Y[2], s = area, c = colors3, alpha = 0.4, label = 'Class C')
    plt.scatter(X[3], Y[3], s = area, c = colors4, alpha = 0.4, label = 'Class D')
    plt.scatter(X[4], Y[4], s = area, c = colors5, alpha = 0.4, label = 'Class E')
    plt.scatter(X[5], Y[5], s = area, c = colors6, alpha = 0.4, label = 'Class F')
    plt.scatter(X[6], Y[6], s = area, c = colors7, alpha = 0.4, label = 'Class G')
    plt.scatter(X[7], Y[7], s = area, c = colors8, alpha = 0.4, label = 'Class H')
    plt.scatter(X[8], Y[8], s = area, c = colors9, alpha = 0.4, label = 'Class I')
    # 画直线，到时候可能不需要
    # plt.plot([0, 9.5], [9.5, 0], linewidth = '0.5', color = '#000000')
    # label的位置
    plt.legend(loc="upper right")
    # 输出路径还要改
    plt.savefig('../../testPic/right.jpg', dpi = 300)
    plt.show()

def fun(srcPath):
    # get the images
    img_list = os.listdir(srcPath)
    x = []
    y = []
    for img in img_list:
        Path = srcPath + '/' + img
        src_img = cv2.imread(Path)
        left_x, left_y, right_x, right_y, x_center, y_center = detector(src_img)
        x.append(right_x)
        y.append(right_y)
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':
    '''
    x1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    y1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    x2 = np.random.normal(7.5, 1.2, 300)
    y2 = np.random.normal(7.5, 1.2, 300)
    X = [x3, x4]
    Y = [y3, y4]
    '''
    X = []
    Y = []
    Path = "C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/back_up"
    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    for dir_name in dir_list:
        srcPath = Path + "/" + dir_name
        # 找到各个类的质心
        x, y = fun(srcPath)
        X.append(x)
        Y.append(y)

    plot(X, Y)
