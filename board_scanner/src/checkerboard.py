import time

import cv2 as cv
import example  # 安装在本地的cpp 五子棋ai
import numpy as np

from src.img2board_matrix import img2board_matrix


class Checkerboard:
    """棋盘类"""

    def __init__(self, matrix: np.ndarray = None):
        self._matrix = matrix
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
        # 判断 ai 执棋颜色,如为黑色则交换黑白（因为之后的判断是基于 ai 执 白）
        if piece_color == 'black':
            for i in range(self.height):
                for j in range(self.width):
                    if self._matrix[i, j] == 1:
                        self._matrix[i,j] = 2
                    elif self._matrix[i, j] == 2:
                        self._matrix[i,j] = 1
        elif piece_color != 'white':
            raise ValueError("Input 'black' or 'white")
        st = time.time()
        mapstring = list()
        for x in range(15):
            mapstring.extend([int(i) for i in self._matrix[x]])
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
        self.cur_step += 1
        return ai_ope_x, ai_ope_y

    def update(self):
        """更新棋盘状态矩阵"""
        pass

    def set_matrix(self, matrix):
        """导入外部矩阵"""
        self._matrix = matrix

    def draw(self, mode):
        """绘制棋盘"""
        pass

    def __str__(self):
        output = ''
        for x in range(self.height):
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
    img_path = "../../archive/image/15_15.png"
    img = cv.imread(img_path)  # matrix(img) size:Width*Height*Dim ,[0,255]
    board_matrix = img2board_matrix(img, 15, 15)
    board = Checkerboard(board_matrix)
    print(board.predict_by_cpp(piece_color='white'))
