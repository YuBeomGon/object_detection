#/usr/bin/bash

git clone https://github.com/tensorflow/models.git
cd models/
git checkout bee6a47121f16e5903463cf3537a3e09f6be42e0

cd research/
# Compile protos.
/home/jhjang/apps/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install .

# Test the installation.
python object_detection/builders/model_builder_tf1_test.py

