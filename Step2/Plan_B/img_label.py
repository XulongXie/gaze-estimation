import os
import multiprocessing
import cv2
from Step2.Plan_B.Eye_Detection import Eye_Detec as detector


def classdef(name):
    if name == '480x270':
        return '0'
    if name == '480x540':
        return '1'
    if name == '480x810':
        return '2'
    if name == '960x270':
        return '3'
    if name == '960x540':
        return '4'
    if name == '960x810':
        return '5'
    if name == '1440x270':
        return '6'
    if name == '1440x540':
        return '7'
    if name == '1440x810':
        return '8'


# label the images
def labelling(src, dst, name):
    # read the file name under the folder
    img_list = os.listdir(src)
    # txt name
    txt_path = dst + '/' + name + '.txt'
    for img in img_list:
        srcPath = src + "/" + img
        image = cv2.imread(srcPath)
        x0, y0, x1, y1, x2, y2, x3, y3 = detector(image)
        with open(txt_path, 'a', encoding='utf-8') as dstFile:
            # src = 'C:/Users/hasee/Desktop/Master Project/Step1/DataSet/480x270'
            src_list = src.split('/')
            class_name = src_list[-1]
            class_type = classdef(class_name)
            text = srcPath
            dstFile.write(text + ' ' + str(x0) + ' ' + str(y0) + ' ' +
                          str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' +
                          str(y2) + ' ' + str(x3) + ' ' + str(y3) + ' ' + class_type)
            dstFile.write('\n')
            dstFile.close()


# main function
if __name__ == '__main__':
    Path = "C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/ValidationSet"
    dstPath = "C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/Label"

    # create a new one if it's not existing
    try:
        os.mkdir(dstPath)
        print(dstPath + ' 创建成功')
    except:
        print("文件夹已经存在，无需创建")

    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    print(dir_list)
    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    print(dir_list)
    dir_name = dir_list[8]
    srcPath = Path + "/" + dir_name
    print(srcPath)
    labelling(srcPath, dstPath, dir_name)
