# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import usr_dir

import sys
import tensorflow as tf
import numpy as np


# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys


class Args(object):

  def __init__(self):
    self.problem = 'translate_enzh_wmt32k'
    self.data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_vocab'
    self.model_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_20190501'
    self.model = 'universal_transformer'
    self.hparams_set = 'universal_transformer_base'


FLAGS = Args()


def translate(encoders, translate_model, ckpt_path, inputs):


  def encode(input_str):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}

  def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
      integers = integers[:integers.index(1)]

    if np.asarray(integers).shape[-1] == 1:
      return encoders["targets"].decode(integers)
    else:
      return encoders["targets"].decode(np.squeeze(integers))

  encoded_inputs = encode(inputs)

  with tfe.restore_variables_on_create(ckpt_path):
    print(tf.global_variables())
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)


def main(_):

  # Fetch the problem
  wmt_problem = problems.problem(FLAGS.problem)

  # Declare the path we need
  data_dir = FLAGS.data_dir

  checkpoint_dir = FLAGS.model_dir
  ckpt_name = FLAGS.problem
  # ckpt_dir = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, ckpt_name))
  ckpt_dir = tf.train.latest_checkpoint(checkpoint_dir)

  # Create hparams and the model
  model_name = FLAGS.model
  hparams_set = FLAGS.hparams_set
  hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name=FLAGS.problem)

  # Get the encoders from the problem
  encoders = wmt_problem.feature_encoders(data_dir)

  translate_model = registry.model(model_name)(hparams, Modes.EVAL)

  sys.stdout.write('> ')
  sys.stdout.flush()
  sentence_en = sys.stdin.readline().strip()
  while sentence_en:
    if sentence_en == 'q':
      print("Close this process")
      break
    outputs = translate(encoders, translate_model, ckpt_dir, sentence_en)
    print(outputs)
    print('> ', end='')
    sys.stdout.flush()
    sentence_en = sys.stdin.readline()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
