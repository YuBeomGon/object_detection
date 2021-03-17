import argparse
import tensorflow as tf


def convert_tflite(saved_model_dir_path, out_path):
    """ Convert exported graph to tflite model
    Args:
        saved_model_dir_path: exported graph directory path
        out_path: converted tflite output file path
    Returns:
        None
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir_path)
    tflite_model = converter.convert()

    # Save the model.
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    
    print("[INFO] Saved TFLite model in {}".format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converting saved model to TFLite model")
    parser.add_argument("--saved_model_dir", type=str, help='exported graph directory path')
    parser.add_argument("--out_tflite_path", type=str, help='converted tflite output file path')

    args = parser.parse_args()

    convert_tflite(args.saved_model_dir, args.out_tflite_path)
