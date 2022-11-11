import copy
from typing import List, Dict, Tuple
import time

import cv2 as cv
import example  # 安装在本地的cpp 五子棋ai
import numpy as np

from board_scanner.src.img2board_matrix import img2board_matrix


class Checkerboard:
    """棋盘类"""

    def __init__(self, matrix: np.ndarray = None):
        self._matrix = matrix       # 矩阵中1代表黑棋，2代表白棋
        self.width = self._matrix.shape[1]
        self.height = self._matrix.shape[0]
        self.cur_step = 1  # 步数
        self.max_search_steps = 3  # 最远搜索2回合之后

    def predict_by_cpp(self, piece_color: str = 'white'):
        """调用 cpp ai 函数进行预测
        Args:
            piece_color: ai该步所走的棋的颜色, 默认ai执白
        Returns:
            ai_ope_x, ai_ope_y: ai建议的下一步坐标

        """
        # 判断 ai 执棋颜色,如为黑色则交换黑白（因为之后的判断是基于默认 ai 执 白（2））
        matrix_temp = copy.deepcopy(self._matrix)
        if piece_color.lower() == 'black':
            for i in range(self.height):
                for j in range(self.width):
                    if matrix_temp[i, j] == 1:
                        matrix_temp[i, j] = 2
                    elif matrix_temp[i, j] == 2:
                        matrix_temp[i, j] = 1
        elif piece_color.lower() != 'white':
            raise ValueError("Input 'black' or 'white")
        st = time.time()
        mapstring = list()
        for x in range(15):
            mapstring.extend([int(i) for i in matrix_temp])
        try:
            node_len, ai_ope_x, ai_ope_y = example.ai_1step(self.cur_step, int(True), self.max_search_steps, mapstring)
            if ai_ope_x == -1 & ai_ope_y == -1:
                raise ValueError('Leaf Node met')
        except ValueError:
            raise ValueError('AI程序计算出来的数值不正确')
        print('Coordinate:%d %d' % (ai_ope_x, ai_ope_y))
        ed = time.time()
        print('生成了%d个节点，用时%.4f' % (node_len, ed - st))
        # self._matrix[ai_ope[0]][ai_ope[1]] = 2
        return ai_ope_x, ai_ope_y

    def update(self, pos: Tuple[int, int], piece_color: str):
        """ 根据落子位置，更新棋盘矩阵状态
        Args:
            pos: 落子位置
            piece_color: 落子的颜色，只可为 'white' 或 'black'
        """
        if not self._matrix[pos[0], pos[1]]:
            raise ValueError("位置 {0} 已经有棋子".format(pos))
        if piece_color.lower() == 'white':
            self._matrix[pos[0], pos[1]] = 2
            self.cur_step += 1
        elif piece_color.lower() == 'black':
            self._matrix[pos[0], pos[1]] = 1
            self.cur_step += 1
        else:
            raise ValueError("参数 piece_color 应为 'black' 或 ‘white’ ")

    def set_matrix(self, matrix):
        """导入外部矩阵"""
        self._matrix = matrix
        self.cur_step = 0

    def draw(self, mode):
        """绘制棋盘"""
        pass

    def __str__(self):
        output = ''
        output += '      0  1  2  3  4  5  6  7  8  9  A  B  C  D  E \n'
        for i,x in enumerate(range(self.height)):
            output += ' {0:3X} '.format(i)
            for y in range(self.width):
                if self._matrix[x][y] == 0:
                    output += ' . '
                elif self._matrix[x][y] == 1:  # 黑棋
                    output += ' ⚪ '
                elif self._matrix[x][y] == 2:  # 白棋
                    output += ' * '
            output += '\n'
        return output


if __name__ == '__main__':
    from src.object_detection import object_detection
    img_path = "../../archive/image/1.png"
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    img = object_detection(img)
    board_matrix = img2board_matrix(img, 15, 15,margin_rate_w=0.008,margin_rate_h=0.008)
    board = Checkerboard(board_matrix)
    print(board.predict_by_cpp(piece_color='white'))
    print(board)
