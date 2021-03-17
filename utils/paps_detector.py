import tensorflow as tf
import numpy as np
import cv2


class PapsmearDetector(object):
    def __init__(self, tflite_model_path):
        self.papsmear_detector =  tf.lite.Interpreter(tflite_model_path)
        self.papsmear_detector.allocate_tensors()
        self.input_details = self.papsmear_detector.get_input_details()
        self.output_details = self.papsmear_detector.get_output_details()

        self.in_idx = self.input_details[0]['index']
        self.out_reg_idx = self.output_details[0]['index']
        self.out_score_idx = self.output_details[1]['index']
        self.out_clf_idx = self.output_details[2]['index']

    def predict_outputs(self, image, input_size=(320, 320), mode='pytorch'):
        input_h, input_w = input_size
        assert input_h == input_w
        input_image = self._preprocess_input(image.astype(np.float32), mode=mode)
        input_image = cv2.resize(input_image, (input_h, input_w))
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

        self.papsmear_detector.set_tensor(self.in_idx, input_image)
        self.papsmear_detector.invoke()

        bboxes = self.papsmear_detector.get_tensor(self.out_reg_idx)
        classes = self.papsmear_detector.get_tensor(self.out_score_idx)
        scores = self.papsmear_detector.get_tensor(self.out_clf_idx)

        bboxes = np.squeeze(bboxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        # Rearrange bboxes
        new_bboxes = np.empty_like(bboxes)
        for idx, bbox in enumerate(bboxes):
            ymin, xmin, ymax, xmax = bbox
            new_bboxes[idx, :] = [xmin, ymin, xmax, ymax]

        return new_bboxes, scores, classes

    def _preprocess_input(self, input_image, mode='pytorch'):
        # if mode == 'pytorch':
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225]

        #     input_image /= 255.0
        #     input_image[:, :, 0] = (input_image[:, :, 0] - mean[0]) / std[0]
        #     input_image[:, :, 1] = (input_image[:, :, 1] - mean[1]) / std[1]
        #     input_image[:, :, 2] = (input_image[:, :, 2] - mean[2]) / std[2]
        
        # else:
        #     mean = 128.0
        #     std = 128.0
        #     input_image = (input_image - mean) / std
        # input_image /= 255.0

        return input_image
