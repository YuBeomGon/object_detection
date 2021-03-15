#!/bin/bash

BASE_URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/"
MODEL = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"


wget ${BASE_URL}${MODEL}.tar.gz \
         -O ${MODEL}.tar.gz

tar -xvf ${MODEL}.tar.gz