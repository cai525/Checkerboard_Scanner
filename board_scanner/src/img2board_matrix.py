import cv2 as cv
import numpy as np

from board_scanner.utils import image_tool

# 定义颜色范围:字典确定的范围内的颜色，将会被认定为指定颜色
white_range = {'r': [170, 255], 'g': [170, 255], 'b': [170, 255]}
black_range = {'r': [0, 80], 'g': [0, 80], 'b': [0, 80]}


def img2board_matrix(img: np.ndarray):
    image_tool.draw_img(img,mode='BGR')
    _,rate = image_tool.color_count(img,white_range,draw_mask=True)
    print(rate)
    # image_tool.draw_hist(img,mode='BGR')


if __name__ == '__main__':
    img_path = "../../archive/image/white_block.jpg"
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    img2board_matrix(img)
