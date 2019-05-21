import tensorflow as tf
import numpy as np

meta_graph_path = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_20190501/model.ckpt-250000.meta'
model_path = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_20190501/'

default_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config=default_config)
load_mod = tf.train.import_meta_graph(meta_graph_path)
load_mod.restore(sess, tf.train.latest_checkpoint(model_path))
all_variables = tf.global_variables()

for _ in range(3):
  with tf.variable_scope('universal_transformer/body/encoder/universal_transformer_basic'):
    print(tf.global_variables())

variable_name_str = ''
for variable_index, variable_item in enumerate(all_variables):
  variable_name_str += variable_item.name + '\n'

with open('/tmp/variable_value.txt', 'w') as f_writer:
  f_writer.write(variable_name_str)
