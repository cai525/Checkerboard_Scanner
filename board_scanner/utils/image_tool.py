import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def draw_img(img: np.ndarray, mode: str):
    """调用matplotlib包，查看矩阵表示的图片"""
    if mode.lower() == 'bgr':
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def draw_hist(img: np.ndarray, mode: str):
    """查看矩阵表示的图片的颜色直方图"""
    if mode.lower() == 'gray':
        plt.hist(img.ravel(), 256, [0, 256])
    elif mode.lower() == 'bgr':
        for i, col in enumerate(('b', 'g', 'r')):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label='col')
            plt.xlim([0, 256])
            plt.legend(loc='best')
    else:
        raise ValueError("Error:Input argument 'mode' must be in 'gray' or 'bgr' ")

    plt.show()


def color_count(img: np.ndarray, color_range: dict, draw_mask: bool):
    """统计图像中指定范围的像素数量
    arg:
        img: 图像矩阵
        color_range: 需要统计的颜色范围 eg:white_range = {'r': [170, 255], 'g': [170, 255], 'b': [170, 255]}
        draw_mask: 是否绘制选定区域
    returns:
        color_num: 对应范围颜色像素个数
        return color_num,color_num/total_num: 对应范围颜色像素占比
    """
    color_num = 0
    total_num = img.size  # 图片像素总数
    if draw_mask:
        mask = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_value = img[i][j]
            if color_range['b'][0] <= pixel_value[0] <= color_range['b'][1] \
                    and color_range['g'][0] <= pixel_value[1] <= color_range['g'][1] \
                    and color_range['r'][0] <= pixel_value[2] <= color_range['r'][1]:
                color_num = (color_num + 1)
                if draw_mask:
                    mask[i][j] = 1

    if draw_mask:
        draw_img(mask,mode='gray')
    return color_num, color_num / total_num


if __name__ == '__main__':
    pass
