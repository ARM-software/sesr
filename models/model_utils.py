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


import tensorflow as tf
import numpy as np


##############################
## LINEAR BLOCK DEFINITIONS ##
##############################


#EXPANDED Linear block
class LinearBlock_e(tf.keras.Model):
    def __init__(self,
                 in_filters: int,
                 num_inner_layers: int,
                 kernel_size: int,
                 padding: str,
                 out_filters: int,
                 feature_size: int):
        super().__init__()
        """
        Expanded linear block. Input --> 3x3 Conv to expand number of channels 
        to 'feature_size' --> 1x1 Conv to project channels into 'out_filters'. 

        At inference time, this can be analytically collapsed into a single, 
        small 3x3 Conv layer. See also the LinearBlock_c class which is a 
        very efficient method to train linear blocks without any loss in 
        image quality.
        """

        def conv2d(filters: int, kernel_size_: int) -> tf.keras.layers.Layer:
            return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size_, padding=padding)

        layers = []
        for _ in range(num_inner_layers):
            layers.extend([conv2d(filters=feature_size, kernel_size_=kernel_size)])
        layers.append(conv2d(filters=out_filters, kernel_size_=1))
        self.block = tf.keras.Sequential(layers)

    def call(self, inputs, training=None, mask=None):
        return self.block(inputs, training=training)


#COLLAPSED Linear block
class LinearBlock_c(tf.keras.Model):
    def __init__(self,
                 in_filters: int,
                 num_inner_layers: int,
                 kernel_size: int,
                 padding: str,
                 out_filters: int,
                 feature_size: int):
        super().__init__()

        """
        This is a simulated linear block in the train path. The idea is to collapse 
        linear block at each training step to speed up the forward pass. The backward 
        pass still updates all the expanded weights. 

        After training is completed, the weight generation ops are replaced by
        a tf.constant at pb/tflite generation time.

        ----------------------------------------------------------------
        |                            padded_identity                   |
        |                                   |                          |
        |                         conv1x1(inCh, r*inCh)  [optional]    |
        |                                   |                          |
        |                        convkxk(r*inCh, r*inCh)               |
        |                                   |                          |
        |                         conv1x1(r*inCh, outCh)               |
        |                                   |                          |
        |  simulating residual: identity -> +                          |
        |         (or) padded_conv1x1_wt    | (weight_tensor generated)|
        ----------------------------------------------------------------
                                            |
                    input_tensor -> Actual convkxk(inCh, outCh)
                                            |
                                        Final output
        """
        def conv2d(filters: int, kernel_size_: int, padding_: str) -> tf.keras.layers.Layer:
            return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size_, padding=padding_)

        # Params
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.feature_size = feature_size

        # If num_inner_layers > 1, then use another conv1x1 at the beginning
        onebyone = True if num_inner_layers > 1 else False

        # expansion with kx,ky kernel and then project to out_filters using 1x1
        kernel_size = [kernel_size, kernel_size]
        self.kx, self.ky = kernel_size

        # shape: (in_filters,in_filters)
        delta = tf.eye(in_filters)

        # expanded shape:(in_filters, 1, 1, in_filters)
        delta = tf.expand_dims(tf.expand_dims(delta, 1), 1)

        # padded shape: (in_filters, kx, ky, in_filters)
        delta = tf.pad(delta, paddings=[[0, 0], [self.kx - 1, self.kx - 1], [self.ky - 1, self.ky - 1], [0, 0]])

        # Ensure the Value isn't trainable
        self.delta = tf.Variable(initial_value=delta, trainable=False, dtype=tf.float32)

        # Learnable Collapse Conv's
        conv1 = conv2d(feature_size, [1, 1], "valid")

        conv2 = conv2d(feature_size, kernel_size, "valid")

        conv3 = conv2d(out_filters, [1, 1], "valid")

        # Define Collapse Block
        if onebyone:
            self.collapse = tf.keras.Sequential([conv1, conv2, conv3])
        else:
            self.collapse = tf.keras.Sequential([conv2, conv3])

        # Calculate Residual
        kernel_dim = [self.kx, self.ky, self.in_filters, self.out_filters]
        residual = np.zeros(kernel_dim, dtype=np.float32)

        if self.in_filters == self.out_filters:
            mid_kx = int(self.kx / 2)
            mid_ky = int(self.ky / 2)

            for out_ch in range(self.out_filters):
                residual[mid_kx, mid_ky, out_ch, out_ch] = 1.0

        # Ensure the Value isn't trainable
        self.residual = tf.Variable(initial_value=residual, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        # Run Through Conv2D's - online linear collapse
        wt_tensor = self.collapse(self.delta)

        # reverse order of elements in 1,2 axes
        wt_tensor = tf.reverse(wt_tensor, tf.constant([1, 2]))

        # (in_filters, kx, ky, out_filters) -> (kx, ky, in_filters, out_filters)
        wt_tensor = tf.transpose(wt_tensor, [1, 2, 0, 3])

        # Direct-residual addition
        # when in_filters != self.out_filters, this is just zeros
        wt_tensor += self.residual

        # Output - the actual conv2d
        out = tf.nn.conv2d(inputs, wt_tensor, strides=[1, 1, 1, 1], padding="SAME")

        return out



