import cv2 as cv
import numpy as np

from board_scanner.utils import image_tool

# 定义颜色范围:字典确定的范围内的颜色，将会被认定为指定颜色
white_range = {'r': [230, 255], 'g': [230, 255], 'b': [230, 255]}
black_range = {'r': [0, 80], 'g': [0, 80], 'b': [0, 80]}
black_hsv = {'lower':np.array([0,0,0]),'upper':np.array([180,255,46])}
white_hsv = {'lower':np.array([0,0,221]),'upper':np.array([180,30,255])}
# 颜色的判定门限——统计某颜色占比高于此门限的区域将被视为指定颜色棋子
threshold = 0.02


# def img2block(img: np.ndarray, width_num: int, height_num: int, margin_rate: float = 0):


def img2board_matrix(img: np.ndarray, width_num: int, height_num: int, margin_rate: float = 0):
    """将照片分为若干子块，每个子块代表棋盘的一个点
    Args:
        img: 待分割图像
        width_num,height_num: 图像的宽边/高边有多少个点位
        margin_rate: 图像的边缘宽占总的宽/高的比
    Returns:
        matrix: 代表棋盘情况的矩阵，0代表该处无棋子，1代表黑棋，2代表白棋
    """
    # 初始化棋盘矩阵
    matrix = np.zeros((height_num, width_num))

    # note: img.shape的第0维为高，第一维为宽
    margin_height, margin_width = img.shape[0] * margin_rate, img.shape[1] * margin_rate,
    img_height, img_width = img.shape[0] - 2 * margin_height, img.shape[1] - 2 * margin_width
    block_height, block_width = img_height / height_num, img_width / width_num  # 每一个block的宽高

    # 遍历每个block
    # stop位置额外减去0.5个block,防止越界
    for i, ph in enumerate(
            np.arange(margin_height, img.shape[0] - margin_height - block_height * 0.5, block_height)):
        for j, pw in enumerate(
                np.arange(margin_width, img.shape[1] - margin_width - block_width * 0.5, block_width)):
            block = img[int(ph):int(ph + block_height), int(pw):int(pw + block_width), :]
            # 统计颜色
            w_rate = image_tool.color_judge(block,white_hsv)
            b_rate = image_tool.color_judge(block,black_hsv)
            # 确定是否有棋子
            if w_rate <= threshold and b_rate <= threshold:
                continue
            elif w_rate > b_rate:
                matrix[i][j] = 2        # 2代表白棋
            else:
                matrix[i][j] = 1        # 1代表黑棋
    # 转成整数
    matrix = matrix.astype(np.int)
    return matrix
    # image_tool.draw_img(img, mode='BGR')

    # image_tool.draw_hist(img,mode='BGR')


if __name__ == '__main__':
    img_path = "../../archive/image/15_15.png"
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    board_matrix = img2board_matrix(img, 15, 15)
    print(board_matrix)
