#/usr/bin/bash

MODEL_DIR=ssd_mobilenet_v2_fpnlite_papsmear
PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
CHECKPOINT_DIR=training/
OUTPUT_DIR=model_exported

# clear old exported model
rm -rf ${OUTPUT_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=2,3 \
    python3 ./models/research/object_detection/export_tflite_graph_tf2.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --trained_checkpoint_dir=${CHECKPOINT_DIR} \
            --output_directory=${OUTPUT_DIR} \
            --max_detections=150

