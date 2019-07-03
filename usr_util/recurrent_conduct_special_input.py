# coding=utf-8

import tensorflow as tf
from tensorflow.python.ops import variable_scope
import numpy as np
import sys
import os
sys.path.append('..')
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry
from tensor2tensor import problems
import functools
from tensor2tensor.models.research import universal_transformer_util
from test_input_generator import positional_encoding_generator


data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_analy/recurret_conduct'


def input_generator():
  return tf.random.normal(shape=(1, 26, 1, 1024), mean=50)


class Args(object):

  def __init__(self):
    self.problem = 'translate_enzh_wmt32k'
    self.data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_vocab'
    self.model_dir = '/home/murphyhuang/dev/mldata/en_ch_translate_output_ut_20190509'
    self.model = 'universal_transformer'
    self.hparams_set = 'adaptive_universal_transformer_base'
    self.ut_type = 'act'
    self.step_num = 1
    self.save_dir = os.path.join(data_dir, 'ut_0509_recurrent_1024_randomonly')


def main():

  FLAGS = Args()

  # Enable TF Eager execution
  tfe = tf.contrib.eager
  tfe.enable_eager_execution()

  batch_inputs = input_generator()

  # initialize translation model
  hparams_set = FLAGS.hparams_set
  Modes = tf.estimator.ModeKeys
  hparams = trainer_lib.create_hparams(hparams_set, data_dir=FLAGS.data_dir, problem_name=FLAGS.problem)
  translate_model = registry.model(FLAGS.model)(hparams, Modes.EVAL)

  # recover parameters and conduct recurrent conduction
  ckpt_dir = tf.train.latest_checkpoint(FLAGS.model_dir)

  with tfe.restore_variables_on_create(ckpt_dir):
    with variable_scope.EagerVariableStore().as_default():
      features = {'inputs': batch_inputs}
      with tf.variable_scope('universal_transformer/body'):
        input_tensor = tf.convert_to_tensor(features['inputs'])
        input_tensor = common_layers.flatten4d3d(input_tensor)
        encoder_input, self_attention_bias, _ = (
          transformer.transformer_prepare_encoder(
            input_tensor, tf.convert_to_tensor([0]), translate_model.hparams, features=None))

      with tf.variable_scope('universal_transformer/body/encoder'):

        ffn_unit = functools.partial(
          universal_transformer_util.transformer_encoder_ffn_unit,
          hparams=translate_model.hparams)

        attention_unit = functools.partial(
          universal_transformer_util.transformer_encoder_attention_unit,
          hparams=translate_model.hparams,
          encoder_self_attention_bias=None,
          attention_dropout_broadcast_dims=[],
          save_weights_to={},
          make_image_summary=True)

      storing_list = []
      transformed_state = encoder_input
      for step_index in range(1024):
        storing_list.append(transformed_state.numpy())

        with tf.variable_scope('universal_transformer/body/encoder/universal_transformer_{}'.format(FLAGS.ut_type)):
          transformed_state = universal_transformer_util.step_preprocess(
            transformed_state,
            tf.convert_to_tensor(step_index % FLAGS.step_num),
            translate_model.hparams
          )
        with tf.variable_scope('universal_transformer/body/encoder/universal_transformer_{}/rec_layer_0'.format(FLAGS.ut_type)):
          transformed_new_state = ffn_unit(attention_unit(transformed_state))
        with tf.variable_scope('universal_transformer/body/encoder'):
          if (step_index + 1) % FLAGS.step_num == 0:
            transformed_new_state = common_layers.layer_preprocess(transformed_new_state, translate_model.hparams)

            if step_index == 5:
              print(transformed_new_state)

        transformed_state = transformed_new_state
      storing_list = np.asarray(storing_list)
      np.save(FLAGS.save_dir, storing_list)


if __name__ == '__main__':
  main()
