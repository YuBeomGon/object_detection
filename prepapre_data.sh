#/usr/bin/bash

mkdir -p data

## Generate Partition and Labels info
python gen_labels_info.py
python gen_partition.py

## Create TFRecords dataset files based on partition and labels info
python gen_tfrecords.py
