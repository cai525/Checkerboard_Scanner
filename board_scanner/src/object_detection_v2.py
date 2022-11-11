import cv2 as cv
import numpy as np

from board_scanner.utils import image_tool

# 确定颜色的上下界
orange_hsv = {'lower': np.array([11, 10, 46]), 'upper': np.array([34, 255, 255])}


def grab_contours(cnts):
    """与cv2.findContours一起使用，返回cnts中的countors(轮廓)

    Args:
        cnts:findContours返回的轮廓

    Returns:

    """
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
                         "otherwise OpenCV changed their cv2.findContours return "
                         "signature yet again. Refer to OpenCV's documentation "
                         "in that case"))

    # return the actual contours array
    return cnts


def find_rectangle(binary: np.ndarray)->np.ndarray:
    """

    Args:
        binary: 二值化后的图片

    Returns:
        box: 最大矩形的四个顶点
    """
    gray = binary * 255  # 变为灰度图像
    edged = cv.Canny(gray, 20, 200, apertureSize=7)  # canny 边缘检测
    # 可选： 采用 闭运算 ，使边界连续（闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象）
    # kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    # edged = cv.dilate(edged, kernel, iterations=2)
    # edged = cv.erode(edged, kernel, iterations=2)

    # image_tool.draw_img(edged, 'GRAY')
    contours = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    # 寻找最大的,且为四边形的轮廓
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:15]

    box = np.zeros(1)
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.048 * peri, True)
        if len(approx) == 4:
            box = approx
            break
    return box.reshape((4,-1))


def object_detection(img: np.ndarray):
    # 高斯模糊，去除噪点
    img = cv.GaussianBlur(img, (5, 5), 1)
    # image_tool.draw_img(img, 'BGR')
    # 根据颜色进行二值化
    mask = image_tool.binary_by_color(img, orange_hsv)
    # 寻找二值化图像后的矩形
    box = find_rectangle(mask)
    # 通过四个顶点，得到棋盘
    return image_tool.perTran(img, box)


if __name__ == '__main__':
    img = cv.imread("../../archive/image/origin_2.png")
    board_img = object_detection(img)
    image_tool.draw_img(board_img,'BGR')
