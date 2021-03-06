#/usr/bin/bash

PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v2_fpnlite_papsmear.config
MODEL_DIR=training/
#MODEL_DIR=backup/

mkdir -p ${MODEL_DIR}_eval
EVAL_DIR=${MODEL_DIR}_eval

# clear old eval results
# rm -rf ${EVAL_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1 python3 ./models/research/object_detection/model_main_tf2.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --checkpoint_dir=${MODEL_DIR} \
            --model_dir=${EVAL_DIR} \
            --run_once \
            --alsologtostderr
