"""
基本图像处理工具库
"""
import cv2
import numpy as np


def crop_image(frame):
    """
    图像裁切为中心的 640*480
    :param frame: 原始图像
    :return: 裁切后的图像
    """
    height, width = frame.shape[:2]
    if width > 640:
        frame = frame[:, int((width - 640) / 2): int((width + 640) / 2)]
    if height > 480:
        frame = frame[int((height - 480) / 2): int((height + 480) / 2), :]
    # 改变分辨率

    return frame


def set_mask(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    为图像设置掩膜
    :param src: 源图像
    :param mask: 掩膜
    :return: 掩膜后图像
    """
    channels = cv2.split(src)
    # 通道分离
    result = []
    for i in range(len(channels)):
        result.append(cv2.bitwise_and(channels[i], mask))
    # 各通道于掩膜进行图像和操作
    dest = cv2.merge(result)
    # 通道合并
    return dest


class ColorFilter(object):
    """
    颜色筛选器类 \n
    筛选出红色的例子 \n
    colorFilter = img_process.ColorFilter() \n
    colorFilter.colorRange = [((0, 43, 46), (10, 255, 255)), ((156, 43, 46), (180, 255, 255))] \n
    red_mask = colorFilter(frame) \n
    """

    def __init__(self):
        self.colorRange = []

    # 这里是颜色筛选的范围
    # 存储格式为 [ ((H起始，S起始，V起始),(H结束，S结束，V结束)), ... ]

    def __call__(self, src: np.ndarray) -> np.ndarray:
        # 必要的函数注释
        finalMask = np.zeros_like(src)[:, :, 0]
        # finalMask指的是 最终合成的掩膜
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # 将图像转为HSV通道
        for each in self.colorRange:
            # 逐一获取HSV范围
            lower, upper = each
            # HSV范围解构为 起始 和 结束
            mask = cv2.inRange(hsv, lower, upper)
            # 制作该HSV范围的掩膜
            finalMask = cv2.bitwise_or(finalMask, mask)
        # 掩膜合并 目标颜色 = 颜色1 + 颜色2 + ... + 颜色n
        # 注：inRange()不在HSV范围内的部分 数值为 0

        dest = set_mask(src, finalMask)
        # 设置掩膜
        return dest


def find_contour_in_roi(frame):
    roi_frame = frame[150:480, 0:640]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, threshold_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('threshold1', threshold_frame)
    # 双边滤波
    # threshold_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('threshold2', threshold_frame)
    # 腐蚀
    kernel = np.ones((6, 6), np.uint8)
    erode_frame = cv2.erode(threshold_frame, kernel, iterations=1)
    # 膨胀
    kernel = np.ones((6, 6), np.uint8)
    dilate_frame = cv2.dilate(255 - erode_frame, kernel, iterations=1)
    # 划定识别区域
    cv2.imshow('line_roi', dilate_frame)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(dilate_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, dilate_frame
