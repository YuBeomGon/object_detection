import numpy as np
from utils import tfrecord_writer


def set_config():
    config = {
        'partition_path': './data/partition.npy',
        'labels_info_path': './data/labels_info.npy',
        'input_shape': (512, 512, 3),
        'data_dir_path': '/home/Dataset/Papsmear/original/',
        'out_dir_path': './data/tfrecords/'
    }
    return config


def generate_tfrecords(config):
    # Load partition and labels info
    partition = np.load(config['partition_path'], allow_pickle=True, encoding='latin1').item()
    labels_info = np.load(config['labels_info_path'], allow_pickle=True, encoding='latin1').item()

    # Initialize TFRecordWriter
    writer = tfrecord_writer.TFRecordWriter(
        partition=partition, 
        labels_info=labels_info,
        input_shape=config['input_shape'], 
        data_dir_path=config['data_dir_path']
    )

    # Write tfrecords file
    writer.write_tfrecords(out_directory=config['out_dir_path'])
    print("[INFO] Finished Creating TFRecords.")


if __name__ == "__main__":
    config = set_config()
    generate_tfrecords(config=config)
