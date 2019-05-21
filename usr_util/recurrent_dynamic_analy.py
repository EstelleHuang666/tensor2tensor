# coding=utf-8
# fft & low-pass filtering

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_analy/recurret_conduct'


def frequency_analy():
  record_storing_path = os.path.join(data_dir, 'ut_0509_recurrent_1024.npy')

  recurrent_record = np.load(record_storing_path)

  for dim_index in range(recurrent_record.shape[3]):
    dim_test = recurrent_record[:, 0, 2, dim_index]
    dim_freq = np.fft.fft(dim_test, dim_test.shape[0])
    dim_freq_abs = np.abs(dim_freq)
    plt.plot(dim_test[:32])
    plt.show()


def euclidean_distance_analy():
  record_storing_path = os.path.join(data_dir, 'ut_0509_recurrent_1024.npy')
  recurrent_record = np.squeeze(np.load(record_storing_path))
  word_index = 12
  recurrent_word_record = recurrent_record[:, word_index, :]
  word_representation_norms = np.linalg.norm(recurrent_word_record, axis=1)
  word_representation_subtract = np.concatenate((np.zeros((1, recurrent_word_record.shape[1])), recurrent_word_record))
  word_representation_subtract = recurrent_word_record - word_representation_subtract[:-1, :]
  word_representation_subtract = word_representation_subtract[1:, :]
  word_representation_euclidean_distance = np.linalg.norm(word_representation_subtract, axis=1)

  word_representation_norms = word_representation_norms.T
  word_representation_euclidean_distance = word_representation_euclidean_distance.T
  ax_1 = plt.subplot(2, 1, 1)
  ax_1.plot(word_representation_norms[1:128])
  ax_1.set_title('Norm of Word Representation through Cycles')
  ax_2 = plt.subplot(2, 1, 2)
  ax_2.plot(word_representation_euclidean_distance[1:128])
  ax_2.set_title('Euclidean Distance between Two Adjacent Cycles')
  plt.show()


def main():
  euclidean_distance_analy()


if __name__ == '__main__':
  main()
