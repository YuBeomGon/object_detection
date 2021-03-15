#/usr/bin/bash

PIPELINE_CONFIG_PATH=configs/

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    python3 ./models/research/object_detection/model_main.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --num_train_steps=20000 \
            --sample_1_of_n_eval_samples=1 \
            --alsologtostderr