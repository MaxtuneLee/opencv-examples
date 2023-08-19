import cv2
import numpy as np
import time


class YOLOv5:
    def __init__(self, model_path, confidenceThehold=0.5, NMSThreshold=0.3):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.confThreshold = confidenceThehold
        self.nmsThreshold = NMSThreshold
        self.classes = ['stop_sign']
        self.frame_count = 0
        self.start = time.time_ns()
        self.fps = -1
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    def detect(self, frame):
        """
        检测图片中的物体
        :param frame: 输入图片
        :return: 检测结果
        """
        new_frame = frame.copy()
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        blob = self.format_yolov5(new_frame)
        self.net.setInput(blob)
        outs = self.net.forward()
        boxes, confs, class_ids = self.unwrap_detection(new_frame, outs[0])
        for (classid, confidence, box) in zip(class_ids, confs, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(new_frame, box, color, 2)
            cv2.rectangle(new_frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(new_frame, self.classes[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if self.frame_count >= 30:
            end = time.time_ns()
            self.fps = 1000000000 * self.frame_count / (end - self.start)
            frame_count = 0
            start = time.time_ns()

        if self.fps > 0:
            fps_label = "FPS: %.2f" % self.fps
            cv2.putText(new_frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return new_frame

    def format_yolov5(self, frame):
        """
        格式化输入图片，将图片转换为网络输入所需的格式
        :param frame: 输入图片
        :return: 格式化后的图片
        """
        # put the image in square big enough
        col, row, _ = frame.shape
        _max = max(col, row)
        resized = np.zeros((_max, _max, 3), np.uint8)
        resized[0:col, 0:row] = frame

        # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
        result = cv2.dnn.blobFromImage(resized, 1 / 255.0, (320, 320), swapRB=True)

        return result

    def unwrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / 640
        y_factor = image_height / 640

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if classes_scores[class_id] > .25:
                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_boxes, result_confidences, result_class_ids
