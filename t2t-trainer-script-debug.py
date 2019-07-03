from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer as t2t_trainer

import sys
import tensorflow as tf

def main(argv):
  t2t_trainer.main(argv)


if __name__ == "__main__":

  # sys_argv = """
  # --data_dir=/home/murphyhuang/dev/mldata/en_ch_translate
  # --problem=translate_enzh_wmt32k
  # --output_dir=/tmp/ut_trial
  # --hparams_set=adaptive_universal_transformer_base_small
  # --model=universal_transformer
  # --schedule=train
  # """

  sys_argv = """
  --data_dir=/home/libi/HDD1/huan1282/dev/mldata/t2t_data/babi_qa
  --problem=babi_qa_concat_task1_1k
  --output_dir=/home/libi/HDD1/huan1282/dev/mldata/t2t_data/babi_qa_output_ut_act_20190703/
  --hparams_set=adaptive_universal_transformer_base
  --model=universal_transformer
  """

  sys_argv = sys_argv.split()
  sys.argv.extend(sys_argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
