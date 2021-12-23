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


from typing import Callable, List, Tuple, Optional
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('scale', 2, 'Scale of SISR')

#Set some dataset related parameters
SCALE = FLAGS.scale
if SCALE != 2 and SCALE != 4:
  raise ValueError('Only x2 or x4 SISR is currently supported')
PATCH_SIZE_HR = 128 if SCALE == 2 else 200
PATCH_SIZE_LR = PATCH_SIZE_HR // SCALE
PATCHES_PER_IMAGE = 64


###########################
## DATASET PREPROCESSING ##
###########################


#Convert RGB image to YCbCr
def rgb_to_ycbcr(rgb: tf.Tensor) -> tf.Tensor:
    ycbcr_from_rgb = tf.constant([[65.481, 128.553, 24.966],
                                  [-37.797, -74.203, 112.0],
                                  [112.0, -93.786, -18.214]])
    rgb = tf.cast(rgb, dtype=tf.dtypes.float32) / 255.
    ycbcr = tf.linalg.matmul(rgb, ycbcr_from_rgb, transpose_b=True)
    return ycbcr + tf.constant([[[16., 128., 128.]]])


#Get the Y-Channel only
def rgb_to_y(example: tfds.features.FeaturesDict) -> Tuple[tf.Tensor, tf.Tensor]:
    lr_ycbcr = rgb_to_ycbcr(example['lr'])
    hr_ycbcr = rgb_to_ycbcr(example['hr'])
    return lr_ycbcr[..., 0:1] / 255., hr_ycbcr[..., 0:1] / 255.


#Extract random patches for training
def random_patch(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    def lr_offset(axis: int):
        size = tf.shape(lr)[axis]
        return tf.random.uniform(shape=(), maxval=size - PATCH_SIZE_LR, 
                                 dtype=tf.dtypes.int32)

    lr_offset_x, lr_offset_y = lr_offset(axis=0), lr_offset(axis=1)
    hr_offset_x, hr_offset_y = SCALE * lr_offset_x, SCALE * lr_offset_y
    lr = lr[lr_offset_x:lr_offset_x + PATCH_SIZE_LR, 
            lr_offset_y:lr_offset_y + PATCH_SIZE_LR]
    hr = hr[hr_offset_x:hr_offset_x + PATCH_SIZE_HR, 
            hr_offset_y:hr_offset_y + PATCH_SIZE_HR]
    return lr, hr


#Data augmentions
def augment(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    u = tf.random.uniform(shape=())
    k = tf.random.uniform(shape=(), maxval=4, dtype=tf.dtypes.int32)

    def augment_(image: tf.Tensor) -> tf.Tensor:
        image = tf.cond(u < 0.5, true_fn=lambda: image, false_fn=lambda: tf.image.flip_up_down(image))
        return tf.image.rot90(image, k=k)

    return augment_(lr), augment_(hr)


#Get many random patches for each image
def patches(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    tuples = (augment(*random_patch(lr, hr)) for _ in range(PATCHES_PER_IMAGE))
    lr, hr = zip(*tuples)
    return list(lr), list(hr)


#Generate INT8 TFLITE
def generate_int8_tflite(model: tf.keras.Model,
                         filename: str,
                         path: Optional[str] = '/tmp',
                         fake_quant: bool = False) -> str:
    saved_model = path + '/' + filename
    model.save(saved_model)
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.inference_type = tf.dtypes.int8
    if fake_quant:  # give some default ranges for activations (for perf-eval only)
        converter.default_ranges_stats = (-6., 6.)
    input_arrays = converter.get_input_arrays()
    # if input node has fake-quant node, then the following ranges would be
    # overridden by fake-quant ranges.
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    if not os.path.exists(path):
        os.makedirs(path)
    tflite_filename = path + '/' + filename + '.tflite'    
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_filename

