from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from tensor2tensor.bin import t2t_decoder

import tensorflow as tf
# Enable TF Eager execution
tf.enable_eager_execution()


def main(argv):
  t2t_decoder.main(argv)


if __name__ == "__main__":
  sys_argv = """
  --data_dir=/home/murphyhuang/dev/mldata/en_ch_translate_vocab
  --output_dir=/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_20190509
  --problem=translate_enzh_wmt32k
  --model=universal_transformer
  --hparams_set=adaptive_universal_transformer_base
  --decode_interactive=True
  """
  sys_argv = sys_argv.split()
  sys.argv.extend(sys_argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
