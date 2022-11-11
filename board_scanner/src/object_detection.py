import numpy as np
import cv2


# import matplotlib.pyplot as plt
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
    return rect  # #


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
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_rectangle(img, L2gradient_find_rectangle):
    """

    Args:
        imge: 输入图像
        L2gradient_find_rectangle：布尔型变量，canny参数
    Returns:
        new_image:找到图像中的最大矩形的图片
    """
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200, apertureSize=7, L2gradient=L2gradient_find_rectangle)  # Perform Edge detection
    ###寻找轮廓
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    ###遮罩,遮去其他无用部分
    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 0), 3)  # (0,0,0)代表边框颜色

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)  # 五子棋棋盘,检测出最大的矩形，还需去掉黑色的部分
    return new_image


def object_detection(img: np.ndarray, L2gradient_find_rectangle=True, L2gradient_wrapped=True) -> np.ndarray:
    """检测输入图片中的棋盘目标
    
    Args:
         img: 包含棋盘的图片
         L2gradient_find_rectangle:布尔型变量，canny参数
         L2gradient_wrapped:布尔型变量，canny参数
    Returns:
        board_img: 检测出的棋盘
    """
    # TODO: 完善该函数
    new_image = find_rectangle(img, L2gradient_find_rectangle)  # 找到最大矩形框
    ###图像矫正
    output = new_image.copy()
    # 转换成灰度图像
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # 双边滤波器能够做到平滑去噪的同时还能够很好的保存边缘
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # 检测边缘
    edged = cv2.Canny(gray, 50, 200, apertureSize=7, L2gradient=L2gradient_wrapped)
    # cv2.imshow("edged",edged)
    # 查找轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    # 获取前3个最大的轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    screenCnt = None
    for c in cnts:
        # 轮廓周长
        peri = cv2.arcLength(c, True)
        print('arcLength : {:.3f}'.format(peri))
        # approxPolyDP主要功能是把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
        # 近似轮廓的多边形曲线， 近似精度为轮廓周长的1.5%
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # 矩形边框具有4个点， 将其他的剔除
        if len(approx) == 4:
            screenCnt = approx
            break
    # 绘制轮廓矩形边框
    cv2.drawContours(new_image, [screenCnt], -1, (0, 255, 0), 3)
    # 调整为x,y 坐标点矩阵
    pts = screenCnt.reshape(4, 2)
    # print('screenCnt.reshape:\n{}'.format(pts))
    # 透视变换
    warped = perTran(output, pts)
    ###从原图片中裁剪
    # 转换换到 HSV（原图是BGR类型，也就是常说的RGB）
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    # 设定c橙色的阀值
    lower_orange = np.array([0, 0, 0])
    upper_orange = np.array([25, 255, 255])
    # 根据阀值构建掩模 （类似有一张黑色的纸，根据上方颜色阀值在纸上扣出橙色的区域）
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    maskInv = cv2.bitwise_not(mask)  # 逆遮罩
    # 对原图像和掩模位运算 （2张图片叠在一起，透过掩膜上的白色洞去看的内容就是结果）
    res_colored = cv2.bitwise_and(warped, warped, mask=mask)  # 扣出棋盘，但是棋子也全部变成了黑色
    # cv2.imwrite("D:/photo_collection/res_colored.png", res_colored)

    ###截取
    frame = cv2.cvtColor(res_colored, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(frame, 100, 255, cv2.THRESH_TRUNC)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    maxarea = 0
    maxint = 0
    for c in contours:
        if cv2.contourArea(c) > maxarea:
            maxarea = cv2.contourArea(c)
            maxint = i
        i += 1

    # 多边形拟合
    epsilon = 0.02 * cv2.arcLength(contours[maxint], True)

    # 多边形拟合
    approx = cv2.approxPolyDP(contours[maxint], epsilon, True)
    [[x1, y1]] = approx[0]
    [[x2, y2]] = approx[2]
    x = np.array([x1, x2])
    y = np.array([y1, y2])
    board_img = warped[np.min(y):np.max(y), np.min(x):np.max(x)]
    return board_img


if __name__ == '__main__':
    # TODO: 在此处测试上述函数是否工作
    # canny参数，TRUE检测边缘减少，FALSE检测边缘增多
    L2gradient_find_rectangle = True  # 寻找最外围轮廓
    L2gradient_wrapped = True  # 仿射变换边缘
    img = cv2.imread("input_image/img_6.png")
    checkerboard = object_detection(img, L2gradient_find_rectangle, L2gradient_wrapped)
    cv2.waitKey(0)
