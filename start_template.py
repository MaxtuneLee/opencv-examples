"""
图像识别初始化模板
"""
import cv2
import numpy as np

from vison.recognition import line_follow
from vison.utils import crop_image

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    if ret:
        cropped_frame = crop_image(frame)
        cv2.imshow('original', cropped_frame)
        line_follow(cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
