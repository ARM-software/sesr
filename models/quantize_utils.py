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


from typing import Callable, List, Tuple, Union
import tensorflow as tf

def compute_ranges(kernel: tf.Tensor, per_channel: bool, symmetric: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    axes = tf.range(tf.rank(kernel) - 1) if per_channel else None
    if symmetric:
        quant_max = tf.stop_gradient(tf.math.reduce_max(tf.math.abs(kernel), axis=axes))
        quant_min = -quant_max
    else:
        quant_max = tf.stop_gradient(tf.math.reduce_max(kernel, axis=axes))
        quant_min = tf.stop_gradient(tf.math.reduce_min(kernel, axis=axes))
    return quant_max, quant_min      

@tf.custom_gradient
def floor_ste(x: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], List[tf.Tensor]]]:
    y = tf.floor(x)

    def grad(dy: tf.Tensor) -> List[tf.Tensor]:
        return [dy]

    return y, grad

def get_nudged_ranges_scale(
        min: tf.Tensor,
        max: tf.Tensor,
        num_bits: int,
        narrow_range: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    quant_max = tf.math.pow(2., tf.cast(num_bits, dtype=tf.dtypes.float32)) - 1.
    quant_min = tf.constant(1.) if narrow_range else tf.constant(0.)
    scale = (max - min) / (quant_max - quant_min)

    # Rounding the zero-point to ensure one of the quantized values snap to zero
    zero_point_from_min = quant_min - min / scale
    nudged_zero_point = tf.round(zero_point_from_min)
    nudged_zero_point = tf.where(zero_point_from_min < quant_min,
                                 quant_min * tf.ones(shape=tf.shape(nudged_zero_point)),
                                 nudged_zero_point)
    nudged_zero_point = tf.where(zero_point_from_min > quant_max,
                                 quant_max * tf.ones(shape=tf.shape(nudged_zero_point)),
                                 nudged_zero_point)

    # adjust/nudge the min/max to ensure zero-point snaps to real zero.
    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale
    return nudged_min, nudged_max, scale

def fake_quant_with_min_max_vars(
        inputs: tf.Tensor,
        min: tf.Tensor,
        max: tf.Tensor,
        num_bits: int,
        narrow_range: bool = False) -> tf.Tensor:
    """
    This is differentiable equivalent of the utility in tf.quantization.

    tf.quantization.fake_quant* utilities only allows the min/max ranges
    to increase through gradients, but we would have to rely on l2_loss
    to decrease the min/max ranges. This updated utility allows the gradients
    to both increase and decrease the min/max ranges.
    """
    nudged_min, nudged_max, scale = get_nudged_ranges_scale(min, max, num_bits, narrow_range)
    clipped_data = tf.clip_by_value(inputs, nudged_min, nudged_max)
    shifted_data = clipped_data - nudged_min
    quant_data = floor_ste(shifted_data / scale + 0.5)
    quant_data = quant_data * scale + nudged_min

    return quant_data

fake_quant_with_min_max_vars_per_channel = fake_quant_with_min_max_vars

class ActivationQuantizationBlock(tf.keras.layers.Layer):
    def __init__(self,
                 enabled: bool,
                 mode: str):
        super().__init__()
        self.enabled = enabled
        self.mode = mode

        if self.mode == 'train':
            self.fake_quant_with_min_max_vars_fn = \
                fake_quant_with_min_max_vars
        elif self.mode == 'infer':
            self.fake_quant_with_min_max_vars_fn = \
                tf.quantization.fake_quant_with_min_max_vars 

    def build(self, input_shape):
        if self.enabled:
            self.quant_min = self.add_weight(
                name='act_quant_min',
                trainable=True)
            self.quant_max = self.add_weight(
                name='act_quant_max',
                trainable=True)
            if self.mode == 'train':
                self.quant_initialized = tf.Variable(False, trainable=False)                   

    def init_quant_ranges(self, inputs: tf.Tensor) -> None:
        quant_max, quant_min = compute_ranges(inputs, per_channel=False, symmetric=False)
        self.quant_max.assign(quant_max)
        self.quant_min.assign(quant_min)
        self.quant_initialized.assign(True)

    def call(self, inputs):
        if self.enabled:
            if self.mode == "train":
                if not self.quant_initialized:
                    self.init_quant_ranges(inputs)

            return self.fake_quant_with_min_max_vars_fn(
                inputs,
                min=self.quant_min,
                max=self.quant_max,
                num_bits=8,
                narrow_range=False)                          
        else:
            return inputs
