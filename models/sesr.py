# Copyright 2021 Arm Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Callable, List, Tuple
import os
import tensorflow as tf
from models import model_utils

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('feature_size', 256, 'Number of features inside linear blocks')
tf.compat.v1.flags.DEFINE_integer('int_features', 16, 'Number of intermediate features within SESR (parameter f in paper)')
tf.compat.v1.flags.DEFINE_integer('m', 5, 'Number of 3x3 layers (parameter m in paper)')
tf.compat.v1.flags.DEFINE_string('linear_block_type', 'collapsed', 'Specify whether to train a linear block which does an online collapsing during training, or a full expanded linear block: Options: "collapsed" [DEFAULT] or "expanded"')


######################
## MODEL DEFINITION ##
######################


#Main SESR network class
class SESR(tf.keras.Model):
    def __init__(self, feature_size: int, m: int, LinearBlock_fn: Callable):
        super().__init__()
        """
          Define a residualFlag that is true if using expanded LinearBlock for 
          training. This will be used in the call function for SESR. If collapsed 
          LinearBlock is used for training, then short residuals are already collapsed 
          within the LinearBlock class.
        """
        self.residualFlag = True if LinearBlock_fn == model_utils.LinearBlock_e else False
        self.input_block = LinearBlock_fn(in_filters = 1,
                                          num_inner_layers=1, 
                                          kernel_size=5, 
                                          padding='same', 
                                          out_filters=FLAGS.int_features,
                                          feature_size=feature_size)
        self.linear_blocks = [
            LinearBlock_fn(in_filters = FLAGS.int_features,
                           num_inner_layers=1, 
                           kernel_size=3, 
                           padding='same', 
                           out_filters=FLAGS.int_features, 
                           feature_size=feature_size)
            for _ in range(m)]
        self.prelus = [tf.keras.layers.PReLU(shared_axes=[1, 2]) for _ in range(m)]
        self.output_block = LinearBlock_fn(in_filters = FLAGS.int_features,
                                           num_inner_layers=1, 
                                           kernel_size=5, 
                                           padding='same', 
                                           out_filters=FLAGS.scale**2,
                                           feature_size=feature_size)

    def call(self, inputs, training=None, mask=None):
        features_0 = features = self.input_block(inputs - 0.5)

        for linear_block, prelu in zip(self.linear_blocks, self.prelus):
          if self.residualFlag:
            features = prelu(linear_block(features) + features)
          else:
            features = prelu(linear_block(features))

        features = self.output_block(features + features_0) + inputs  #residual across whole network
        features = tf.nn.depth_to_space(features, block_size=2)
        if FLAGS.scale == 4: #Another depth_to_space if scale == 4
          features = tf.nn.depth_to_space(features, block_size=2)
        return tf.clip_by_value(features, clip_value_min=0., clip_value_max=1.)


