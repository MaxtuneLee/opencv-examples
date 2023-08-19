import cv2
import numpy as np
from lzn_modules.vison.utils import find_contour_in_roi
from lzn_modules.vison.utils import set_mask, ColorFilter


def detect_split_road(frame, contours):
    """
    三岔路口巡线识别
    :param frame: 一帧图像
    :return: 中心偏移量
    """
    split_road_flag = 0
    if len(contours) > 1:
        # print('数量：', len(contours))
        # 找到面积最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 计算面积
        area = cv2.contourArea(max_contour)
        # print('area size: ', area)
        split_road_flag = 1
        for contour in contours:
            single_area = cv2.contourArea(contour)
            # print(single_area)
            if single_area < 3000.0 or single_area > 5000.0:
                split_road_flag = 0
                break
        # if split_road_flag == 1:
        #     print('split road')

    return split_road_flag


def detect_large_object_in_roi(frame, contours, object_area):
    """
        大块物体检测
        :param object_area: 识别面积大小阈值
        :param contours: 视觉内轮廓
        :param frame: 一帧图像
        :return: 中心偏移量
    """
    big_blank_flag = 0
    if len(contours) >= 1:
        # print('数量：', len(contours))
        # 找到面积最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 计算面积
        area = cv2.contourArea(max_contour)
        print('large area size: ', area)
        if area > object_area:
            big_blank_flag = 1
        # if big_blank_flag == 1:
        #     print('big-blank')

    return big_blank_flag


def blackFilter(frame):
    """
    扣出黑色区域并进行二值化操作
    :param frame:
    :return:
    """
    colorFilter = ColorFilter()
    colorFilter.colorRange = ([((0, 0, 0), (255, 255, 70))])
    black_mask = colorFilter(frame)
    # 创建全白的frame
    white_frame = np.ones_like(frame) * 255
    # 将黑色区域进行合成
    added = cv2.addWeighted(white_frame, 0.2, black_mask, 1, 0.8)
    # cv2.imshow('blackFilter', added)
    return added