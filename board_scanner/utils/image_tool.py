from typing import List, Tuple, Dict
import cv2 as cv
import numpy as np


def binary_by_color(img: np.ndarray, hsv_range: Dict[str, np.ndarray]):
    """根据颜色对图像进行二值化"""
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转为 hsv
    # 开运算：先腐蚀后膨胀，用于移除由图像噪音形成的斑点
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    hsv_img = cv.erode(hsv_img, kernel, iterations=1)
    mask = cv.inRange(hsv_img, hsv_range['lower'], hsv_range['upper'])  # 计算掩膜
    # 膨胀
    mask = cv.dilate(mask, kernel, iterations=1)
    # draw_img(mask, 'GRAY')
    return mask


def draw_img(img: np.ndarray, mode: str):
    import matplotlib.pyplot as plt
    """调用matplotlib包，查看矩阵表示的图片"""
    if mode.lower() == 'bgr':
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def draw_hsv_hist(img: np.ndarray):
    import matplotlib.pyplot as plt
    """查看矩阵表示的图片的颜色直方图"""
    # H
    hist = cv.calcHist([img], [0], None, [180], [0, 180])
    plt.subplot(3, 1, 1)
    plt.plot(hist, label='H')
    plt.xlim([0, 180])
    plt.legend(loc=2)
    # S
    hist = cv.calcHist([img], [1], None, [255], [0, 255])
    plt.subplot(3, 1, 2)
    plt.plot(hist, label='S')
    plt.xlim([0, 255])
    plt.legend(loc=2)
    # V
    hist = cv.calcHist([img], [2], None, [255], [0, 255])
    plt.subplot(3, 1, 3)
    plt.plot(hist, label='V')
    plt.xlim([0, 255])
    plt.legend(loc=2)

    plt.show()


def color_judge(img: np.ndarray, hsv_range):
    """使用hsv判断色块颜色

    Args:
        img:图像矩阵
        hsv_range:颜色的hsv范围
    """
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    inRange_hsv = cv.inRange(hsv_img, hsv_range['lower'], hsv_range['upper']) // 255
    # draw_img(inRange_hsv,'gray')
    return inRange_hsv.sum() / img.size


def detect_circle(img: np.ndarray, min_r: int) -> bool:
    """检测是否有圆形"""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 1000, param1=50, param2=11, minRadius=min_r, maxRadius=1000)
    if circles is None:
        return False
    # for circle in circles[0]:
    #     if circle[2] >= 100:
    #         continue
    #     # 坐标行列
    #     x = int(circle[0])
    #     y = int(circle[1])
    #     # 半径
    #     r = int(circle[2])
    #
    #     # 在原图用指定颜色标记出圆的位置
    #     img = cv.circle(img, (x, y), r, (0, 0, 255), -1)
    #     draw_img(img,'BGR')
    return True


def order_points(pts):
    """矩形的四个顶点"""
    rect = np.zeros((4, 2), dtype='float32')
    # 坐标点求和 x+y
    s = pts.sum(axis=1)
    # np.argmin(s) 返回最小值在s中的序号
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # diff就是后一个元素减去前一个元素  y-x
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回矩形有序的4个坐标点
    return rect


def perTran(image, pts):
    """

    Args:
        image: 输入图片
        pts: 检测到的矩形的四个点

    Returns:
        warped: 仿射变换矫正后的图片
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 定义变换后新图像的尺寸
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype='float32')
    # 变换矩阵
    M = cv.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


if __name__ == '__main__':
    img_path = "../../archive/image/white_block1.jpg"
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    black_hsv = {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 46])}
    white_hsv = {'lower': np.array([0, 0, 221]), 'upper': np.array([180, 30, 255])}
    rate = color_judge(img, white_hsv)
    print(rate)
