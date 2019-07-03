from tensorflow.python.util import function_utils
from tensorflow.python.framework import ops
from tensorflow.python.training import training


class _DatasetInitializerHook(training.SessionRunHook):
  """Creates a SessionRunHook that initializes the passed iterator."""

  def __init__(self, iterator):
    self._iterator = iterator

  def begin(self):
    self._initializer = self._iterator.initializer

  def after_create_session(self, session, coord):
    del coord
    session.run(self._initializer)


def _get_features_and_labels_from_input_fn(input_fn, mode, params, config):
  """Extracts the `features` and labels from return values of `input_fn`."""
  return parse_input_fn_result(_call_input_fn(input_fn, mode, params, config))


def _call_input_fn(input_fn, mode, params, config):
  """Calls the input function.
  Args:
    input_fn: The input function.
    mode: `tf.estimator.ModeKeys`
  Returns:
    The return value of the passed `input_fn`, which should be one of:
      * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
          tuple `(features, labels)` with same constraints as below.
      * A tuple `(features, labels)`: Where `features` is a `Tensor` or a
        dictionary of string feature name to `Tensor` and `labels` is a
        `Tensor` or a dictionary of string label name to `Tensor`. Both
        `features` and `labels` are consumed by `model_fn`. They should
        satisfy the expectation of `model_fn` from inputs.
  Raises:
    ValueError: if `input_fn` takes invalid arguments.
  """
  input_fn_args = function_utils.fn_args(input_fn)
  kwargs = {}
  if 'mode' in input_fn_args:
    kwargs['mode'] = mode
  if 'params' in input_fn_args:
    kwargs['params'] = params
  if 'config' in input_fn_args:
    kwargs['config'] = config
  with ops.device('/cpu:0'):
    return input_fn(**kwargs)


def parse_input_fn_result(result):
  """Gets features, labels, and hooks from the result of an Estimator input_fn.
  Args:
    result: output of an input_fn to an estimator, which should be one of:
      * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
          tuple (features, labels) with same constraints as below.
      * A tuple (features, labels): Where `features` is a `Tensor` or a
        dictionary of string feature name to `Tensor` and `labels` is a
        `Tensor` or a dictionary of string label name to `Tensor`. Both
        `features` and `labels` are consumed by `model_fn`. They should
        satisfy the expectation of `model_fn` from inputs.
  Returns:
    Tuple of features, labels, and input_hooks, where features are as described
    above, labels are as described above or None, and input_hooks are a list
    of SessionRunHooks to be included when running.
  Raises:
    ValueError: if the result is a list or tuple of length != 2.
  """
  input_hooks = []
  try:
    # We can't just check whether this is a tf.data.Dataset instance here,
    # as this is plausibly a PerDeviceDataset. Try treating as a dataset first.
    iterator = result.make_initializable_iterator()
  except AttributeError:
    # Not a dataset or dataset-like-object. Move along.
    pass
  else:
    input_hooks.append(_DatasetInitializerHook(iterator))
    result = iterator.get_next()
  return parse_iterator_result(result) + (input_hooks,)


def parse_iterator_result(result):
  """Gets features, labels from result."""
  if isinstance(result, (list, tuple)):
    if len(result) != 2:
      raise ValueError(
          'input_fn should return (features, labels) as a len 2 tuple.')
    return result[0], result[1]
  return result, None
