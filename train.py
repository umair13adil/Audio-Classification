import argparse
import os
from glob import glob

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from models import Conv1D, Conv2D, LSTM


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, 1, int(self.sr * self.dt)), dtype=np.int16)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            # print('Label: {}'.format(label))
            # print('Classes: {}'.format(self.n_classes))
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(1, -1)
            Y[i,] = to_categorical(label - 1, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    src_root = args.src_root

    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'SR': sr,
              'DT': dt}
    models = {'conv1d': Conv1D(**params),
              'conv2d': Conv2D(**params),
              'lstm': LSTM(**params)}
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    print(wav_paths)
    print(labels)

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.5,
                                                                  random_state=1)
    tg = DataGenerator(wav_train, label_train, sr, dt,
                       len(set(label_train)), batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       len(set(label_val)), batch_size=batch_size)

    model = models[model_type]
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=30, verbose=1,
              callbacks=[csv_logger, cp])
    print(model.summary())

    # Create a converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True

    # Convert the model
    tflite_model = converter.convert()

    # Create the tflite model file
    tflite_model_name = "android/app/src/main/assets/converted_model.tflite"
    open(tflite_model_name, "wb").write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='lstm',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='clean',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)
