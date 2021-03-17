#/usr/bin/bash

PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v2_fpnlite_papsmear.config
OUT_MODEL_DIR=training/

mkdir -p ${OUT_MODEL_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1 python3 ./models/research/object_detection/model_main_tf2.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	        --model_dir=${OUT_MODEL_DIR}
            --sample_1_of_n_eval_samples=1 \
            --alsologtostderr
