#/usr/bin/bash

MODEL_DIR=ssd_mobilenet_v2_fpnlite_papsmear
PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
CHECKPOINT_PREFIX=${MODEL_DIR}/ckpt-21
OUTPUT_DIR=model_exported

# clear old exported model
rm -rf ${OUTPUT_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1 \
    python3 ./models/research/object_detection/export_inference_graph.py \
            --input_type=image_tensor \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
            --output_directory=${OUTPUT_DIR}