"""
这边放置识别类代码
"""
from typing import Literal

import cv2
import numpy as np
from lzn_modules.vison.utils import find_contour_in_roi


def line_follow(frame, contours, mode):
    """
    巡线识别
    :param frame: 一帧图像
    :return: 中心偏移量
    """
    new_frame = frame.copy()
    if len(contours) != 0:
        print(mode)
        if mode:
            # 计算面积和中间的偏移距离，找到左边的轮廓
            for contour in contours:
                thisMoment = cv2.moments(contour)
                if thisMoment['m00'] != 0:
                    cx = int(thisMoment['m10'] / thisMoment['m00'])
                    cy = int(thisMoment['m01'] / thisMoment['m00'])
                    if cx < 320:
                        offset = cx - 320
                        print('该左拐了')
                        return offset
        # 找到面积最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 计算最大轮廓的中心点
        M = cv2.moments(max_contour)
        cv2.drawContours(new_frame, contours, -1, (0, 255, 0), 3)
        offset = 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 在原图中标记中心点
            cv2.circle(new_frame, (cx, cy + 150), 5, (0, 0, 255), -1)
            cv2.imshow('line_follow', new_frame)
            # 计算中心点与图像中心的偏移量
            offset = cx - 320
        # print(offset)
        # # 根据偏移量控制小车转向
        # if offset > 50:
        #     print('right')
        # elif offset < -50:
        #     print('left')
        # else:
        #     print('forward')
        return offset


def detect_rectangle(frame):
    """
    检测矩形
    :param frame: 一帧图像
    :return: 1有矩形，0无矩形
    """

    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    squares = []
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin = cv2.Canny(gray, 30, 100, apertureSize=3)
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.4 and 10900 < cv2.contourArea(cnt) < 71250:
                print('cnt_area: ' + str(cv2.contourArea(cnt)))
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                cv2.putText(frame, ("#%d" % index), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                squares.append(cnt)
    cv2.drawContours(frame, squares, -1, (0, 255, 0), 3)
    # cv2.imshow('square', frame)
    return len(squares) > 0


def detect_circle(frame):
    """
    检测圆
    :param frame: 一帧图像
    :return: 1有圆，0无圆
    """
    # 转换到灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    eroded = cv2.erode(binary, kernel)
    # cv2.imshow("eroded", eroded)
    # 霍夫圆变换
    circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT, 2, 640, param1=38, param2=70, minRadius=60, maxRadius=650)
    # 输出返回值，方便查看类型
    # print(circles)
    if circles is not None:
        # 输出检测到圆的个数
        # print(len(circles[0]))
        # 根据检测到圆的信息，画出每一个圆
        for circle in circles[0]:
            # 圆的基本信息
            # print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])
            print('cirle: ' + str(circle[2]))
            # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
            frame = cv2.circle(frame, (x, y), r, (0, 0, 255), 3, 8, 0)
    # 显示新图像
    # cv2.imshow('Result', frame)
    return circles is not None


def detect_line(frame):
    """
    检测直线
    :return: 是否有直线存在，直线总体相交的角度
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    # print(lines)
    angle = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            # 计算一下这条直线的斜率
            k = (y2 - y1) / (x2 - x1)
            # 获取下一条直线的斜率
            if i + 1 >= len(lines):
                break
            k1 = (lines[i + 1][0][3] - lines[i + 1][0][1]) / (lines[i + 1][0][2] - lines[i + 1][0][0])
            # 计算两条直线的夹角
            angle = np.arctan(np.abs((k - k1) / (1 + k * k1))) * 180 / np.pi
            # print('angle: ' + str(angle))
    # cv2.imshow('Result', frame)
    return lines is not None, angle
