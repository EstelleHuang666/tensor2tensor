#!/usr/bin/env python
"""t2t-decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_decoder

import tensorflow as tf

def main(argv):
  t2t_decoder.main(argv)


if __name__ == "__main__":
  import pdb; pdb.set_trace()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
