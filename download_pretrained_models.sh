#!/bin/bash

BASE_URL="http://download.tensorflow.org/models/object_detection/"
MODEL="ssdlite_mobilenet_v2_coco_2018_05_09"

wget --no-check-certificate \
         ${BASE_URL}${MODEL}.tar.gz \
         -O /tmp/${MODEL}.tar.gz

tar xzvf /tmp/${model}.tar.gz