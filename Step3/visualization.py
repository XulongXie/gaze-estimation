import os
import cv2
import win32gui
import win32con
import win32api
import time
import colorsys
def posdef(classes):
    if classes == 0:
        return (480, 270)
    if classes == 1:
        return (480, 540)
    if classes == 2:
        return (480, 810)
    if classes == 3:
        return (960, 270)
    if classes == 4:
        return (960, 540)
    if classes == 5:
        return (960, 810)
    if classes == 6:
        return (1440, 270)
    if classes == 7:
        return (1440, 540)
    if classes == 8:
        return (1440, 810)

def visual(pred):
    cord = posdef(pred)
    return cord

# 画框
def drawRect(pos, pred, colors):
    hwnd = win32gui.GetDesktopWindow()
    # 定义框颜色
    hPen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(colors[pred][0], colors[pred][1], colors[pred][2]))
    win32gui.InvalidateRect(hwnd, None, True)
    win32gui.UpdateWindow(hwnd)
    win32gui.RedrawWindow(hwnd, None, None,
                          win32con.RDW_FRAME | win32con.RDW_INVALIDATE | win32con.RDW_UPDATENOW | win32con.RDW_ALLCHILDREN)
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetDC(hwnd)

    win32gui.SelectObject(hwndDC, hPen)
    # 定义透明画刷，这个很重要！
    hbrush = win32gui.GetStockObject(win32con.NULL_BRUSH)
    prebrush = win32gui.SelectObject(hwndDC, hbrush)
    # 左上到右下的坐标
    win32gui.Rectangle(hwndDC, pos[0] - 240, pos[1] - 135, pos[0] + 240, pos[1] + 135)
    win32gui.SaveDC(hwndDC)
    win32gui.SelectObject(hwndDC, prebrush)
    # 回收资源
    win32gui.DeleteObject(hPen)
    win32gui.DeleteObject(hbrush)
    win32gui.DeleteObject(prebrush)
    win32gui.ReleaseDC(hwnd, hwndDC)

if __name__ == '__main__':
    hsv_tuples = [(x / 8, 1., 1.)
                  for x in range(8)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    pred = 1
    cord = visual(pred)
    while True:
        drawRect(cord, pred, colors)

