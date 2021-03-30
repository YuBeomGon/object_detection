# papsmear-object-det-api

## Environment
- Python: 3.8.3
- Tensorflow: 2.4.1

## How to run
1. Install "Tensorflow Object Detection API"

    ```./tf_models_install.sh```

2. Install related libraries

    ```pip install -r requirements.txt```

3. Prepare supplementary files for training model
    - partition.npy
    - labels_info.npy
    - train / test TFRecords files

    ```./prepare_data.sh```

4. Train Model
    
    ```./train.sh```

5. Convert the trained model

    ```./export_graph.sh```

    ```./export_tflite.sh```
