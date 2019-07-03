# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import sys
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import

from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import decoding
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.usr_debug import estimator_util
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config


flags = tf.flags
FLAGS = flags.FLAGS

# See utils/flags.py for additional command-line flags.
flags.DEFINE_string("t2t_usr_dir", None,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_string("tpu_job_name", None,
                    "TPU job name. TPUEstimator can auto-infer this but if the "
                    "configuration is esoteric it should be provided here.")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
flags.DEFINE_bool("use_tpu_estimator", False, "Whether to use TPUEstimator. "
                  "This is always enabled when use_tpu is True.")
flags.DEFINE_bool("xla_compile", False,
                  "Whether to use XLA to compile model_fn.")
flags.DEFINE_integer("xla_jit_level", -1,
                     "GlobalJitLevel to use while compiling the full graph.")
flags.DEFINE_integer("tpu_infeed_sleep_secs", None,
                     "How long to sleep the infeed thread.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_bool("profile", False, "Profile performance?")
flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                     "Number of inter_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                     "Number of intra_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
# TODO(lukaszkaiser): resolve memory and variable assign issues and set to True.
flags.DEFINE_bool(
    "optionally_use_dist_strat", False,
    "Whether to use TensorFlow DistributionStrategy instead of explicitly "
    "replicating the model. DistributionStrategy is used only if the "
    "model replication configuration is supported by the DistributionStrategy.")
# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_string("schedule", "continuous_train_and_eval",
                      "Method of Experiment to run.")
  flags.DEFINE_integer("eval_steps", 100,
                       "Number of steps in evaluation. By default, eval will "
                       "stop after eval_steps or when it runs through the eval "
                       "dataset once in full, whichever comes first, so this "
                       "can be a very large number.")
except:  # pylint: disable=bare-except
  pass

flags.DEFINE_string("std_server_protocol", "grpc",
                    "Protocol for tf.train.Server.")

# Google Cloud TPUs
flags.DEFINE_string("cloud_tpu_name", "%s-tpu" % os.getenv("USER"),
                    "Name of Cloud TPU instance to use or create.")

# Google Cloud ML Engine
flags.DEFINE_bool("cloud_mlengine", False,
                  "Whether to launch on Cloud ML Engine.")
flags.DEFINE_string("cloud_mlengine_master_type", None,
                    "Machine type for master on Cloud ML Engine. "
                    "If provided, overrides default selections based on "
                    "--worker_gpu. User is responsible for ensuring "
                    "type is valid and that --worker_gpu matches number of "
                    "GPUs on machine type. See documentation: "
                    "https://cloud.google.com/ml-engine/reference/rest/v1/"
                    "projects.jobs#traininginput")
# Hyperparameter tuning on Cloud ML Engine
# Pass an --hparams_range to enable
flags.DEFINE_string("autotune_objective", None,
                    "TensorBoard metric name to optimize.")
flags.DEFINE_bool("autotune_maximize", True,
                  "Whether to maximize (vs. minimize) autotune_objective.")
flags.DEFINE_integer("autotune_max_trials", 10,
                     "Maximum number of tuning experiments to run.")
flags.DEFINE_integer("autotune_parallel_trials", 1,
                     "How many trials to run in parallel (will spin up this "
                     "many jobs.")
# Note than in open-source TensorFlow, the dash gets converted to an underscore,
# so access is FLAGS.job_dir.
flags.DEFINE_string("job-dir", None,
                    "DO NOT USE. Exists only for Cloud ML Engine to pass in "
                    "during hyperparameter tuning. Overrides --output_dir.")
flags.DEFINE_integer("log_step_count_steps", 100,
                     "Number of local steps after which progress is printed "
                     "out")


def set_hparams_from_args(args):
  """Set hparams overrides from unparsed args list."""
  if not args:
    return

  hp_prefix = "--hp_"
  tf.logging.info("Found unparsed command-line arguments. Checking if any "
                  "start with %s and interpreting those as hparams "
                  "settings.", hp_prefix)

  pairs = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith(hp_prefix):
      pairs.append((arg[len(hp_prefix):], args[i+1]))
      i += 2
    else:
      tf.logging.warn("Found unknown flag: %s", arg)
      i += 1

  as_hparams = ",".join(["%s=%s" % (key, val) for key, val in pairs])
  if FLAGS.hparams:
    as_hparams = "," + as_hparams
  FLAGS.hparams += as_hparams


def create_hparams():
  hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams,
                                    hparams_path=hparams_path)


def create_run_config(hp, output_dir=None):
  """Create a run config.

  Args:
    hp: model hyperparameters
    output_dir: model's output directory, defaults to output_dir flag.

  Returns:
    a run config
  """
  save_ckpt_steps = max(FLAGS.iterations_per_loop, FLAGS.local_eval_frequency)
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert FLAGS.output_dir or FLAGS.checkpoint_path
  tpu_config_extra_kwargs = {}
  if FLAGS.tpu_job_name is not None:
    tpu_config_extra_kwargs["tpu_job_name"] = FLAGS.tpu_job_name

  if getattr(hp, "mtf_mode", False):
    save_ckpt_steps = None  # Disable the default saver
    save_ckpt_secs = None  # Disable the default saver
    tpu_config_extra_kwargs = {
        "num_cores_per_replica": 1,
        "per_host_input_for_training": tpu_config.InputPipelineConfig.BROADCAST,
    }

  # the various custom getters we have written do not play well together yet.
  # TODO(noam): ask rsepassi for help here.
  daisy_chain_variables = (
      hp.daisy_chain_variables and
      hp.activation_dtype == "float32" and
      hp.weight_dtype == "float32")
  return trainer_lib.create_run_config(
      model_name=FLAGS.model,
      model_dir=output_dir or os.path.expanduser(FLAGS.output_dir),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=save_ckpt_steps,
      save_checkpoints_secs=save_ckpt_secs,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.enable_graph_rewriter,
      use_tpu=FLAGS.use_tpu,
      use_tpu_estimator=FLAGS.use_tpu_estimator,
      xla_jit_level=FLAGS.xla_jit_level,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      optionally_use_dist_strat=FLAGS.optionally_use_dist_strat,
      daisy_chain_variables=daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs,
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      log_step_count_steps=FLAGS.log_step_count_steps,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
      tpu_config_extra_kwargs=tpu_config_extra_kwargs,
      cloud_tpu_name=FLAGS.cloud_tpu_name)


def main(argv):
  set_hparams_from_args(argv[1:])
  hparams = create_hparams()
  hparams.add_hparam("data_dir", FLAGS.data_dir)
  trainer_lib.set_random_seed(FLAGS.random_seed)

  hparams_lib.add_problem_hparams(hparams, FLAGS.problem)
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)

  eval_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                                  hparams,
                                                  dataset_kwargs={'dataset_split': None})

  features, labels, input_hooks = estimator_util._get_features_and_labels_from_input_fn(
    train_input_fn, tf.estimator.ModeKeys.TRAIN, hparams, create_run_config(hparams))

  print(features)
  print(labels)
  print(input_hooks)
