import tkinter as tk
import time
import threading
from multiprocessing import Value
import cv2
import os

from Step2.preProcessing.PreProcess import GrayImg, lightRemove, gamma_trans, letter_box

import numpy as np
import math


# The moving area
delta_x = 960
delta_y = 540
m = 0
n = 0


# global value to stop/start process
alive = Value('b', True)
flag = False


# Click the button then the window will move
def moveit():
    global delta_x, delta_y, flag, m, n
    window2 = tk.Tk()
    # On the top
    window2.attributes("-topmost", True)
    window2.overrideredirect(True)
    # Fill with red
    Full_color = tk.Label(window2, bg='red', width=10, height=10)
    Full_color.pack()
    n = 0
    # from(300,150)，steps(330. 195)，20 times in total
    while(n <= delta_y):
        m = 0
        while(m <= delta_x):
            window2.geometry("%dx%d+%d+%d" % (10, 10, 480 + m, 270 + n))
            #print("(%d, %d)" %(m, n))
            # Update the window
            window2.update()
            # put the signal as true
            flag = True
            # every 60 secs
            time.sleep(40)
            m = m + 480
        n = n + 270
    flag = False
    time.sleep(5)
    window2.destroy()


'''
def moveit_run():
    t1 = threading.Thread(target = moveit)
    t1.start()
'''

def Operation():
    # Build windows
    window1 = tk.Tk()

    # Find my resolution
    w = window1.winfo_screenwidth()
    h = window1.winfo_screenheight()
    # Print the resolution of my screen
    print(w)
    print(h)

    # Set window 1
    window1.title('Background')
    window1.geometry("%dx%d+%d+%d" % (100, 50, w / 2 - 50, h - 100))
    # window1.attributes("-alpha",0.5)
    window1.overrideredirect(True)
    # Create a button
    button = tk.Button(window1, text='move', bg='yellow', font=('Arial', 16), width=20, height=3, command=moveit)
    # Pack it down
    button.pack()
    window1.mainloop()

def catchPhoto():
    global m, n, flag
    # Output path for image capturing
    base_dir = "C:/Users/hasee/Desktop/Master_Project/Step1/ValidationSet"
    cap = cv2.VideoCapture(0)
    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
    # set window size
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # variable for img name
    i = 6
    while True:
        ret, frame = cap.read()
        # mirror
        frame = cv2.flip(frame, 1)
        first_point = (128, 80)
        last_point = (512, 460)
        cv2.rectangle(frame, first_point, last_point, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        dir_name = ("%dx%d" % (480 + m, 270 + n))
        output_dir = base_dir + "/" + dir_name
        key = cv2.waitKey(1)
        if key == 13:
            """Press Enter to capture image"""
            # create a new dir if it not exist
            try:
                os.mkdir(output_dir)
                i = 6
            except:
                pass
            # Gray value image
            img_gray = GrayImg(frame)
            # calculate the mean value
            mean = np.mean(img_gray)
            # adaptive gamma
            gamma_val = math.log10(0.5) / math.log10(mean / 255)
            # gamma transfer
            image_gamma = gamma_trans(frame, gamma_val)
            # back to gray-level
            image_gamma_correct = GrayImg(image_gamma)
            # light move
            img_gamma_Remove = lightRemove(image_gamma_correct)
            crop = img_gamma_Remove[80:460, 128:512]
            crop = letter_box(crop, [224, 224])
            output_path = os.path.join(output_dir, "%04d.jpg" % i)
            cv2.imwrite(output_path, crop)
            i += 1
        '''
        dir_name = ("%dx%d" % (300 + m, 150 + n))
        output_dir = base_dir + "/" + dir_name
        print(output_dir)
        # create a new dir if it not exist
        try:
            os.mkdir(output_dir)
        except:
            pass
        output_path = os.path.join(output_dir, "%04d.jpg" % i)
        cv2.imwrite(output_path, frame)
        i += 1
        '''
        if key == 82:
            """Press r to rest the index"""
            i = 6
        if key == 27:
            """Press ESC to exit"""
            break
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # Create a queue
    '''
    pcQueue = multiprocessing.Queue()
    point_Process = multiprocessing.Process(target = Operation, args=(pcQueue,))
    point_Process.daemon = True
    camera_Process = multiprocessing.Process(target = catchPhoto, args=(pcQueue,))
    camera_Process.daemon = True
    '''

    # pcQueue = multiprocessing.Queue()
    point_thread = threading.Thread(target=Operation)
    point_thread.daemon = True
    camera_thread = threading.Thread(target=catchPhoto)
    camera_thread.daemon = True
    # start multi-processing
    point_thread.start()
    camera_thread.start()
    # wait a certain time the kill the main process
    time.sleep(600)



