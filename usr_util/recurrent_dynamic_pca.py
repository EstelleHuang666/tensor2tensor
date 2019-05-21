# coding=utf-8
# pca & visualization

import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from sklearn.decomposition import PCA


data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_analy/recurret_conduct'

def space_dynamic_plot(data_mat):
  cmap = plt.get_cmap('viridis')
  cs = np.arange(0, data_mat.shape[0])
  cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(data_mat[:, 0], data_mat[:, 1], data_mat[:, 2], c=scalarMap.to_rgba(cs))
  scalarMap.set_array(cs)
  fig.colorbar(scalarMap)
  plt.show()
  input()


def single_sentence_main():
  record_storing_path = os.path.join(data_dir, 'ut_0509_recurrent_1024.npy')

  recurrent_record = np.squeeze(np.load(record_storing_path))

  start_index = 10
  record_current_time = recurrent_record[start_index, :-1, :]
  pca_tmp_handler = PCA(n_components=3)
  pca_tmp_handler.fit(record_current_time)
  print(pca_tmp_handler.explained_variance_ratio_)

  for step_index in range(start_index, recurrent_record.shape[0]):
    record_current_time = recurrent_record[step_index, :-1, :]
    # pca_tmp_handler = PCA(n_components=3)
    # pca_tmp_handler.fit(record_current_time)
    # print(pca_tmp_handler.explained_variance_ratio_)
    record_reduced_dim = pca_tmp_handler.transform(record_current_time)
    space_dynamic_plot(record_reduced_dim)


def multi_sentence_main():
  record_storing_path_1 = os.path.join(data_dir, 'ut_0509_recurrent_1024.npy')
  recurrent_record_1 = np.squeeze(np.load(record_storing_path_1))
  record_storing_path_2 = os.path.join(data_dir, 'ut_0509_recurrent_1024_jabberwocky.npy')
  recurrent_record_2 = np.squeeze(np.load(record_storing_path_2))

  step_index = 1023
  record_current_time_1 = recurrent_record_1[step_index, :-1, :]
  record_current_time_2 = recurrent_record_2[step_index, :-1, :]
  record_current_time = np.concatenate((record_current_time_1, record_current_time_2))
  pca_tmp_handler = PCA(n_components=3)
  pca_tmp_handler.fit(record_current_time)
  print(pca_tmp_handler.explained_variance_ratio_)
  record_reduced_dim_1 = pca_tmp_handler.transform(record_current_time_1)
  record_reduced_dim_2 = pca_tmp_handler.transform(record_current_time_2)

  space_dynamic_plot(record_reduced_dim_2)
  space_dynamic_plot(np.concatenate((record_reduced_dim_2, record_reduced_dim_1)))


def main():
  single_sentence_main()


if __name__ == '__main__':
  main()
