import cv2 as cv
import numpy as np

from board_scanner.utils import image_tool
# 定义颜色范围:字典确定的范围内的颜色，将会被认定为指定颜色
black_hsv = {'lower':np.array([0,0,0]),'upper':np.array([180,255,86])}
white_hsv = {'lower':np.array([60,0,170]),'upper':np.array([180,20,255])}
# # 颜色的判定门限——统计某颜色占比高于此门限的区域将被视为指定颜色棋子
threshold = 0.01


# def img2block(img: np.ndarray, width_num: int, height_num: int, margin_rate: float = 0):


def img2board_matrix(img: np.ndarray, width_num: int, height_num: int, margin_rate_w: float = 0,margin_rate_h: float = 0):
    """将照片分为若干子块，每个子块代表棋盘的一个点
    Args:
        img: 待分割图像
        width_num,height_num: 图像的宽边/高边有多少个点位
        margin_rate_h: 图像的边缘高占总的高的比
        margin_rate_w: 图像的边缘宽占总的高的比
    Returns:
        matrix: 代表棋盘情况的矩阵，0代表该处无棋子，1代表黑棋，2代表白棋
    """
    # 初始化棋盘矩阵
    matrix = np.zeros((height_num, width_num))

    # note: img.shape的第0维为高，第一维为宽
    margin_height, margin_width = img.shape[0] * margin_rate_h, img.shape[1] * margin_rate_w,
    img_height, img_width = img.shape[0] - 2 * margin_height, img.shape[1] - 2 * margin_width
    block_height, block_width = img_height / height_num, img_width / width_num  # 每一个block的宽高

    # 遍历每个block
    # stop位置额外减去0.5个block,防止越界
    for i, ph in enumerate(
            np.arange(margin_height, img.shape[0] - margin_height - block_height * 0.5, block_height)):
        for j, pw in enumerate(
                np.arange(margin_width, img.shape[1] - margin_width - block_width * 0.5, block_width)):
            block = img[int(ph):int(ph + block_height), int(pw):int(pw + block_width), :]
            # if i == 7 and j == 7:
            #     print(1)
            #     image_tool.draw_img(block, 'bgr')

            # 检测是否存在圆
            if not image_tool.detect_circle(block,int((block_width+block_height)//5)):
                continue
            # 统计颜色
            w_rate = image_tool.color_judge(block,white_hsv)
            b_rate = image_tool.color_judge(block,black_hsv)
            # 再次确定是否有棋子
            if w_rate <= threshold and b_rate <= threshold:
                continue
            image_tool.draw_img(block, 'bgr')
            if b_rate > w_rate:
                matrix[i][j] = 1  # 1代表黑棋
                # image_tool.draw_img(block, 'bgr')
            else:
                matrix[i][j] = 2        # 2代表白棋
                # image_tool.draw_img(block, 'bgr')

    # 转成整数
    matrix = matrix.astype(int)
    return matrix


if __name__ == '__main__':
    from board_scanner.src.object_detection_v2 import object_detection
    img_path = "../../archive/image/origin_2.png"
    # 读出原始图片
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    # 棋盘检测
    img = object_detection(img)
    # image_tool.draw_img(img,'bgr')
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    image_tool.draw_img(hsv_img,mode='')
    board_matrix = img2board_matrix(img, 15, 15,margin_rate_w=0.01,margin_rate_h=0.01)
    print(board_matrix)
