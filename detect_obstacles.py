from yolov4.tf import YOLOv4
import tensorflow as tf
import cv2
import numpy as np

class YoloOD:
    def __init__(self, tiny_model= True):
        self.tiny_model= tiny_model

        if self.tiny_model:
            self.yolo = YOLOv4(tiny=True)
            self.yolo.classes = "yolov4/coco.names"
            self.yolo.make_model()
            self.yolo.load_weights("yolov4/yolov4-tiny.weights", weights_type="yolo")

        else:
            self.yolo = YOLOv4(tiny=False)
            self.yolo.classes = "yolov4/coco.names"
            self.yolo.make_model()
            self.yolo.load_weights("yolov4/yolov4.weights", weights_type="yolo")

    def run_obstacle_detection(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = self.yolo.resize_image(img)
        resized_image = resized_image / 255.
        input_data = resized_image[np.newaxis, ...].astype(np.float32)

        candidates = self.yolo.model.predict(input_data)

        _candidates = []
        result = img.copy()
        for candidate in candidates:
            batch_size = candidate.shape[0]
            grid_size = candidate.shape[1]
            _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
            candidates = np.concatenate(_candidates, axis=1)
            pred_bboxes = self.yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
            pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)]
            pred_bboxes = self.yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
            result = self.yolo.draw_bboxes(img, pred_bboxes)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result, pred_bboxes