import io
import PIL.Image as pil
import os
import cv2
import tensorflow as tf
import numpy as np

from utils import transformations


class TFRecordWriter(object):
    def __init__(self, partition, labels_info, 
                    data_dir_path="./data/", input_shape=(512, 512, 3)):
        """ TFRecord Writer Init.
        Args:
            partition: partition dict
            labels_info: labels_info dict
            data_dir_path: dataset directory path
            input_shape: input shape
        """
        # Supplementary - partition and labels info
        self.partition = partition
        self.labels_info = labels_info
        # Dataset directory path
        self.data_dir_path = data_dir_path
        # Input size
        self.input_shape = input_shape
        # Classes
        self.class_mapper = {
            'Normal': 'Normal',
            'Benign': 'Normal',
            'ASCUS': 'Low-Risk',
            'LSIL': 'High-Risk',
            'HSIL': 'High-Risk',
            'Carcinoma': 'High-Risk',
        }
        self.classes = ['Normal', 'Low-Risk', 'High-Risk']

    def write_tfrecords(self, out_directory):
        if not os.path.exists(out_directory):
            os.mkdir(out_directory)

        train_IDs = self.partition['train']
        test_IDs = self.partition['test']

        print("Num of Trainset: {}".format(len(train_IDs)))
        print("Num of Testset: {}".format(len(test_IDs)))

        # Trainset
        print("[INFO] Start writing train tfrecords")
        out_path = out_directory + "trainset_papsmear.tfrecords"
        tfrecord_writer = tf.io.TFRecordWriter(out_path)
        # Collect train data
        for idx, ID in enumerate(train_IDs):
            features = self._get_data(ID)
            tfrecord_writer.write(features.SerializeToString())
        tfrecord_writer.close()
        print("[INFO] Finished saving train tfrecord files in {}".format(out_directory))

        # Testset
        print("[INFO] Start writing test tfrecords")
        out_path = out_directory + "testset_papsmear.tfrecords"
        tfrecord_writer = tf.io.TFRecordWriter(out_path)
        # Collect test data
        for idx, ID in enumerate(test_IDs):
            features = self._get_data(ID)
            tfrecord_writer.write(features.SerializeToString())
        tfrecord_writer.close()
        print("[INFO] Finished saving test tfrecord files in {}".format(out_directory))

    def _get_data(self, data_ID):
        # Load X and Y
        img_path = self.data_dir_path + data_ID
        with tf.io.gfile.GFile(img_path, 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        img = pil.open(encoded_png_io)
        img = np.asarray(img)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = self.labels_info.get(data_ID)
        ## Crop and transform
        # img, labels = self._crop_and_transform_data(img, labels)

        height, width = img.shape[:2]
        img_format = 'jpg'
        
        xmins = [label[1] for label in labels]
        ymins = [label[2] for label in labels]
        xmaxs = [label[3] for label in labels]
        ymaxs = [label[4] for label in labels]

        classes_text = [self.class_mapper.get(label[0]) for label in labels]
        classes = [self._class_text_to_int(text) for text in classes_text]

        # Encoding string
        classes_text = [text.encode('utf-8') for text in classes_text]
        data_ID = data_ID.encode('utf-8')
        img_format = img_format.encode('utf-8')

        features = {
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/filename': self._bytes_feature(data_ID),
            'image/source_id': self._bytes_feature(data_ID),
            'image/encoded': self._bytes_feature(encoded_png),
            'image/format': self._bytes_feature(img_format),
            'image/object/bbox/xmin': self._float_list_feature(xmins),
            'image/object/bbox/xmax': self._float_list_feature(xmaxs),
            'image/object/bbox/ymin': self._float_list_feature(ymins),
            'image/object/bbox/ymax': self._float_list_feature(ymaxs),
            'image/object/class/text': self._bytes_list_feature(classes_text),
            'image/object/class/label': self._int64_list_feature(classes),
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _crop_and_transform_data(self, img, labels):
        # crop image
        img = transformations.crop_image(img)
        height, width = img.shape[:2]
        # and transform bboxes
        new_labels = []
        for idx, label in enumerate(labels):
            cname, xmin, ymin, xmax, ymax = label
            bbox_point = [xmin, ymin, xmax, ymax]
            new_bbox_point = transformations.transform_bbox_points(img, bbox_point)
            new_label = [
                cname, 
                new_bbox_point[0] / width,
                new_bbox_point[1] / height, 
                new_bbox_point[2] / width,
                new_bbox_point[3] / height
            ]
            new_labels.append(new_label)
        
        return img, new_labels
        
    def _class_text_to_int(self, class_text):
        if class_text == self.classes[0]:
            return 1
        elif class_text == self.classes[1]:
            return 2
        elif class_text == self.classes[2]:
            return 3

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
