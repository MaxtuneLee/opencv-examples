"""
图像识别初始化模板
"""
import math
import cv2
import pytesseract

from lzn_modules.vison.recognition import line_follow, detect_line, detect_rectangle
from lzn_modules.vison.utils import crop_image
from visons import detect_split_road, detect_large_object_in_roi, blackFilter
from lzn_modules.communication.gpio_serial import SerialPort
import lzn_modules.vison.yolov5 as YOLO
import lzn_modules.vison.fastdet as FASTDET
from lzn_modules.vison.nanodet import my_nanodet
from lzn_modules.vison.utils import find_contour_in_roi
from lzn_modules.main_module import MainModule
import time

# 一些基本配置
module = MainModule("dev", computer=True, lineFollow=False, largeObject=False, splitRoad=False, yolo=True,
                    serial=False)

if not module.computer:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 30)

once_large_object_flag = 0
split_flag = 0

if module.serial:
    mySerial = SerialPort('/dev/ttyTHS1', 115200, 1)

if module.yolo:
    # net = YOLO.YOLOv5('yolov5s.onnx')
    net = my_nanodet('nanodet.onnx')
    # net = FASTDET.FastestDet(classesPath='datasets/stop_sign/stop_sign.names',network='stop_sign.onnx')

while 1:
    # if mySerial.isOpen():
    #     # theStatus = mySerial.send_data_async(bytearray([dataToSend]))
    #     # if theStatus == 'fulfilled':
    #     #     dataToSend += 1
    #     mySerial.receive_data()
    #     if once_large_object_flag:
    #         mySerial.send_data_async(bytearray([0x53]))
    # mySerial.receive_data()
    ret, frame = cap.read()
    if ret:
        start = time.time()
        cropped_frame = crop_image(frame)
        if module.env == 'dev':
            cv2.imshow('original', cropped_frame)
        # blackFiltered = blackFilter(cropped_frame)
        contours, bnwFrame = find_contour_in_roi(cropped_frame)
        # frame_count = cap.get(cv2.CAP_PROP_FPS)
        # print('frame_count: ', frame_count)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 以时间戳为文件名保存图片
            cv2.imwrite('photosets/' + str(time.time()) + '.jpg', frame)
            print('photosets/' + str(time.time()) + '.jpg saved')
        if module.yolo:
            sign = net.detect(cropped_frame)
            cv2.imshow('sign', sign)
        if module.largeObject:
            object_flag = detect_large_object_in_roi(frame, contours, 18000.0)
            if object_flag:
                once_large_object_flag = 1
        if module.splitRoad:
            split_flag = detect_split_road(cropped_frame, contours)
        # print('split_flag: ', split_flag, 'once_large_object_flag: ', once_large_object_flag)
        if module.lineFollow:
            center_offset = line_follow(cropped_frame, contours, split_flag and once_large_object_flag)
            if center_offset is not None:
                # print(center_offset)
                data_to_send = int(math.floor(((center_offset + 320) / 640) * 256))
                # print(data_to_send)
                if module.serial:
                    mySerial.send_data(bytearray([data_to_send]))
        if split_flag and once_large_object_flag:
            module.mPrint('real split road')
            if module.serial:
                mySerial.send_data_async(bytearray([0x73]))
            once_large_object_flag = 0
        # ret, angle = detect_line(cropped_frame)
        # if (ret):
        #     print(angle)
        end = time.time()
        # Time elapsed
        seconds = end - start
        # print("Time taken : {0} seconds".format(seconds))
        # Calculate frames per second
        fps = 1 / seconds
        # module.mPrint("Estimated frames per second : {0}".format(fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
