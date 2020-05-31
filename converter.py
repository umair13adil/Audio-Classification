import tensorflow as tf
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow_core.python.keras.models import load_model
import argparse


def convert(args):
    model = load_model(args.model_fn,
                       custom_objects={'Melspectrogram': Melspectrogram,
                                       'Normalization2D': Normalization2D})
    print(model.summary())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops=True
    tflite_model = converter.convert()
    open("android/app/src/main/assets/converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    convert(args)
