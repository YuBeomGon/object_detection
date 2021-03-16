#/usr/bin/bash

PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v2_fpnlite_papsmear.config

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1 python3 ./models/research/object_detection/model_main_tf2.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	    --model_dir=./training/
            --num_train_steps=20000 \
            --sample_1_of_n_eval_samples=1 \
            --alsologtostderr
