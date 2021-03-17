from absl import flags
import tensorflow as tf

flags.DEFINE_string('saved_model_dir', None, 'exported graph directory path')
flags.DEFINE_string('out_tflite_path', None, 'converted tflite output file path')

FLAGS = flags.FLAGS


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
    convert_tflite(FLAGS.saved_model_dir, FLAGS.out_tflite_path)
