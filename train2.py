import argparse
import os
from glob import glob

import librosa
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from models2 import compileModel
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
    params = {'SR': sr,
              'DT': dt}
    model = compileModel(**params)
    print(model.summary())

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    print(wav_paths)

    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    print(labels)

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.8,
                                                                  random_state=0)
    tg = DataGenerator(wav_train, label_train, sr, dt,
                       len(set(label_train)), batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       len(set(label_val)), batch_size=batch_size)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    history = model.fit(tg, epochs=100, callbacks=[es, mc], validation_data=vg)
    # history = model.fit(np.array(wav_train),np.array(label_train), epochs=100, callbacks=[es, mc], validation_data=(wav_val, label_val))

    from matplotlib import pyplot
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)
