import cv2
import numpy as np
import math

# Gray-Level
def GrayImg (img):
    # openCV inner function
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = img.shape
    print(shape)
    my_gray = np.zeros((shape[0], shape[1]), dtype = np.float32)
    # loop for every pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            # attention: G,B,R
            # I(x,y)) = 0.7152G + 0.0722B + 0.2126R
            my_gray[i, j] = 0.7152 * img[i, j, 0]  + 0.0722 * img[i, j, 1] + 0.2126 * img[i, j, 2]
    # convert to uint8
    my_gray = my_gray.astype(np.uint8)

    return gray_cv, my_gray

# calculate the threshold
def CalculateM (img):
    # calculate the row
    row = img.shape[0]
    upper_row = row // 2
    print("upper_row = " + str(upper_row))
    pixel_list = []

    for i in range(0, upper_row):
        for j in range(0, img.shape[1]):
            pixel_list.append(img[i, j])

    # sort the list
    pixel_list.sort()
    # select the median gray-level value
    length = len(pixel_list)
    print(length)
    if (length % 2 != 0):
        index = length // 2
        value = pixel_list[int(index)]
    else:
        index1 = (length - 1) // 2
        index2 = length / 2
        value = int((int(pixel_list[int(index1)]) + int(pixel_list[int(index2)])) / 2)

    return value

# remove the light influence
def lightRemove (img, M):
    afterRemove = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i, j] <= M):
                afterRemove[i, j] = img[i, j]
            else:
                afterRemove[i, j] = M

    afterRemove = afterRemove.astype(np.uint8)
    return afterRemove

# binarization
def Binarization (img):
    # initialization
    img_new = img.astype(np.float32)
    S_N = 0
    P_N = 0
    for i in range(1, img_new.shape[0] - 1):
        for j in range(1, img_new.shape[1] - 1):
            S_N += max(abs(img_new[i - 1, j] - img_new[i + 1, j]), abs(img_new[i, j - 1] - img_new[i, j + 1]))
            P_N += img_new[i, j] * max(abs(img_new[i - 1, j] - img_new[i + 1, j]), abs(img_new[i, j -1] - img_new[i, j + 1]))
    T = P_N / S_N
    T = int(T)
    # binarization
    ret, thresh1 = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)

    return  thresh1, thresh2

# black hat operation
def blackHat (img, kernel_size):
    # set the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # apply top-hat function
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    return black

# top hat operation
def topHat (img, kernel_size):
    # set the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # apply top-hat function
    top = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    return top

# function to calculate Gamma
def gamma_cal(img_gray):
    # to make sure it's a gray-level image
    assert (len(img_gray.shape) == 2)
    # calculation the mean value
    mean = np.mean(img_gray)
    # Gamma = lg(0.5) / lg(mean / 255)
    gamma_adaptive = math.log10(0.5) / math.log10(mean / 255)
    gamma_list = [1 / 4, 1 / 3, 1 / 2.2, 1, 1.8, 2.2, 4.0]
    return gamma_list, gamma_adaptive

# Gamma correction operation
def gamma_trans(img):
    # get the gamma value
    gamma_list, gamma_adaptive = gamma_cal(img)
    # build a gamma table to reduce the complexity
    gamma_table1 = [np.power(x / 255.0, gamma_adaptive) * 255.0 for x in range(256)]   # adaptive value
    # transfer the gray value as integer
    gamma_table1 = np.round(np.array(gamma_table1)).astype(np.uint8)
    # create other gamma tables
    gamma_table_list = []
    for gamma in gamma_list:
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        gamma_table_list.append(gamma_table)
    # append the first gamma table
    gamma_table_list.append(gamma_table1)
    # transfer the images
    img_list = []
    for gamma_tables in gamma_table_list:
        # according to the Gamma table to find the pixel value
        img_list.append(cv2.LUT(img, gamma_tables))

    return img_list


if __name__ == '__main__':

    img = cv2.imread('C:/Users/hasee/Desktop/Master_Project/testPic/overlap.jpg')
    # original
    cv2.imshow("img",img)
    gray1, gray2 = GrayImg(img)
    M_value = CalculateM(gray2)
    cv2.imshow("gray img1", gray1)
    cv2.imshow("gray img2", gray2)
    # after remove the light influence
    newGray = lightRemove(gray2, M_value)
    cv2.imshow("after remove the light influence", newGray)
    # gamma operation
    img_list = gamma_trans(gray2)
    list = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(img_list)):
        cv2.imshow("gamma"+str(list[i]), img_list[i])
    '''
    # gray-level image
    cv2.imshow("gray img1", gray1)
    cv2.imshow("gray img2", gray2)
    # after remove the light influence
    newGray = lightRemove(gray2, M_value)
    cv2.imshow("after remove the light influence", newGray)
    # use the black hat operation
    black = blackHat(newGray, 35)
    cv2.imshow("black hat", black)
    # use the top hat operation
    top = topHat(newGray, 35)
    cv2.imshow("top hat", top)
    # binarization
    thresh1, thresh2 = Binarization(gray1)
    # cv2.imshow("binarization1", thresh1)
    # cv2.imshow("binarization2", thresh2)
    '''
    cv2.waitKey(0)