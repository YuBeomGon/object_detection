#/usr/bin/bash

MODEL_DIR=ssd_mobilenet_v2_fpnlite_papsmear
PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
CHECKPOINT_PREFIX=training/ckpt-21
OUTPUT_DIR=model_exported

# clear old exported model
rm -rf ${OUTPUT_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1 \
    python3 ./models/research/object_detection/export_tflite_ssd_graph.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
            --output_directory=${OUTPUT_DIR} \
            --max_detections=150 \
            --max_classes_per_detection=3 \
            --detections_per_class=300
