import io
import os
import cv2
import tensorflow as tf
import numpy as np
import six
import contextlib2

from utils import transformations

class TFRecordWriter(object):
    def __init__(self, partition, labels_info, 
                    data_dir_path="./data/", input_shape=(320, 320, 3)):
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
            'ASCUS': 'Abnormal',
            'LSIL': 'Abnormal',
            'HSIL': 'Abnormal',
            'Carcinoma': 'Abnormal',
        }
#         self.classes = ['Normal', 'Abnormal']
        self.classes = ['Abnormal']

    def write_tfrecords(self, out_directory):
        if not os.path.exists(out_directory):
            os.mkdir(out_directory)

        train_IDs = self.partition['train']
        val_IDs = self.partition['val']
        test_IDs = self.partition['test']

        print("Num of Trainset: {}".format(len(train_IDs)))
        print("Num of Valset: {}".format(len(val_IDs)))
        print("Num of Testset: {}".format(len(test_IDs)))

        # Trainset
        print("[INFO] Start writing train tfrecords")
        out_path = out_directory + "trainset_papsmear.tfrecords"
        tfrecord_writer = tf.io.TFRecordWriter(out_path)
        # Collect train data
        for idx, ID in enumerate(train_IDs):
            features = self._get_data(ID)
            tfrecord_writer.write(features.SerializeToString())

            # For Verbose
            if idx % 1000 == 0:
                print(idx, ID)

        tfrecord_writer.close()
        print("[INFO] Finished saving train tfrecord files in {}".format(out_directory))
        
        # valset
        print("[INFO] Start writing val tfrecords")
        out_path = out_directory + "valset_papsmear.tfrecords"
        tfrecord_writer = tf.io.TFRecordWriter(out_path)
        # Collect train data
        for idx, ID in enumerate(val_IDs):
            features = self._get_data(ID)
            tfrecord_writer.write(features.SerializeToString())

            # For Verbose
            if idx % 1000 == 0:
                print(idx, ID)

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
            
            # For Verbose
            if idx % 1000 == 0:
                print(idx, ID)
                
        tfrecord_writer.close()
        print("[INFO] Finished saving test tfrecord files in {}".format(out_directory))

    def _get_data(self, data_ID):
        # Load X and Y
        img_path = self.data_dir_path + data_ID
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = self.labels_info.get(data_ID)
        
        ## Preprocess image and bboxes
        img, labels = self._crop_and_transform_data(img, labels)

        ## Preprocess input image
#         input_img = self._preprocess_input(img, mean=128.0, std=128.0)
        input_img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
#         input_img = input_img / 255.0

        height, width = input_img.shape[:2]
        img_format = data_ID.split('.')[-1]
#         img_format = 'jpg'
        
        xmins = [label[1] for label in labels]
        ymins = [label[2] for label in labels]
        xmaxs = [label[3] for label in labels]
        ymaxs = [label[4] for label in labels]

        classes_text = [self.class_mapper.get(label[0]) for label in labels]
        classes = [self._class_text_to_int(text) for text in classes_text]

        # Encoding
        classes_text = [text.encode('utf-8') for text in classes_text]
        data_ID = data_ID.encode('utf-8')
        img_format = img_format.encode('utf-8')
        # encoded_img = input_img.tostring()
        cv2.imwrite("./_temp.jpg", input_img)
        with tf.io.gfile.GFile("./_temp.jpg", 'rb') as fid:
            encoded_img = fid.read()

        features = {
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/filename': self._bytes_feature(data_ID),
            'image/source_id': self._bytes_feature(data_ID),
            'image/encoded': self._bytes_feature(encoded_img),
            'image/format': self._bytes_feature(img_format),
            'image/object/bbox/xmin': self._float_list_feature(xmins),
            'image/object/bbox/xmax': self._float_list_feature(xmaxs),
            'image/object/bbox/ymin': self._float_list_feature(ymins),
            'image/object/bbox/ymax': self._float_list_feature(ymaxs),
            'image/object/class/text': self._bytes_list_feature(classes_text),
            'image/object/class/label': self._int64_list_feature(classes),
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _preprocess_input(self, crop_img, mean=128.0, std=128.0):
        """ Resize and normalize input
        """
        input_img = cv2.resize(crop_img, (self.input_shape[0], self.input_shape[1]))
        input_img = (input_img - mean) / std
        
        return input_img

    def _crop_and_transform_data(self, img, labels):
        # switch image
        img = transformations.switch_image(img)
        
        # and transform bboxes
        labels_list =[]
        bbox_points = []
        for idx, label in enumerate(labels):
            cname, xmin, ymin, xmax, ymax = label
            bbox_point = [xmin, ymin, xmax, ymax]
            if xmax > xmin and ymax > ymin :
                labels_list.append(cname)
                bbox_points.append(bbox_point)

        transformed = transformations.transforms(image=img, bboxes=bbox_points, labels=labels_list)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['labels']    
        height, width = transformed_image.shape[:2]
        
        new_labels = []
        for label, bbox in zip(transformed_class_labels, transformed_bboxes) :
            new_label = [
                label, 
                bbox[0] / width,
                bbox[1] / height, 
                bbox[2] / width,
                bbox[3] / height
            ]
            new_labels.append(new_label)
            self.bbox_check(new_label)
        
        return transformed_image, new_labels
    
    def bbox_check(self, labels) :
        for point in labels[1:] :
            if point < 0 or point > 1.0 :
                raise BoundingBOXError
        
        
    def _class_text_to_int(self, class_text):
        if class_text == self.classes[0]:
            return 1
        elif class_text == self.classes[1]:
            return 2

    def _open_sharded_output_tfrecords(self, exit_stack, base_path, num_shards):
        """Opens all TFRecord shards for writing and adds them to an exit stack.

        Args:
            exit_stack: A context2.ExitStack used to automatically closed the TFRecords
            opened in this function.
            base_path: The base path for all shards
            num_shards: The number of shards

        Returns:
            The list of opened TFRecords. Position k in the list corresponds to shard k.
        """
        tf_record_output_filenames = [
            '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
            for idx in six.moves.range(num_shards)
        ]

        tfrecords = [
            exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
            for file_name in tf_record_output_filenames
        ]

        return tfrecords

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
