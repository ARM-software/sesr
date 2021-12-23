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
import tensorflow_datasets as tfds
import atexit

from models import sesr, model_utils

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')
tf.compat.v1.flags.DEFINE_integer('batch_size', 32, 'Batch size during training')
tf.compat.v1.flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for ADAM')
tf.compat.v1.flags.DEFINE_string('model_name', 'SESR', 'Name of the model')
tf.compat.v1.flags.DEFINE_bool('quant_W', False, 'Quantize weights')
tf.compat.v1.flags.DEFINE_bool('quant_A', False, 'Quantize activations')
tf.compat.v1.flags.DEFINE_bool('gen_tflite', False, 'Generate TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_height', 1080, 'Height of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_width', 1920, 'Width of LR image in TFLITE')

import utils

#Set some dataset processing parameters and some save/load paths
DATASET_NAME = 'div2k' if FLAGS.scale == 2 else 'div2k/bicubic_x4'
if not os.path.exists('logs/'):
  os.makedirs('logs/')
BASE_SAVE_DIR = 'logs/x2_models/' if FLAGS.scale == 2 else 'logs/x4_models/'
if not os.path.exists(BASE_SAVE_DIR):
  os.makedirs(BASE_SAVE_DIR)

SUFFIX = 'QAT' if (FLAGS.quant_W and FLAGS.quant_A) else 'FP32'

if FLAGS.scale == 4: #Specify path to load x2 models (x4 SISR will only finetune x2 models)
  if FLAGS.model_name == 'SESR':
    PATH_2X = 'logs/x2_models/'+FLAGS.model_name+'_m{}_f{}_x2_fs{}_{}Training_{}'.format(
                                                                  FLAGS.m, 
                                                                  FLAGS.int_features,
                                                                  FLAGS.feature_size,
                                                                  FLAGS.linear_block_type, 
                                                                  SUFFIX)


##################################
## TRAINING AND EVALUATION LOOP ##
##################################


def main(unused_argv):

    data_dir = os.getenv("TFDS_DATA_DIR", None)

    dataset_train, dataset_validation = tfds.load(DATASET_NAME, 
                                   split=['train', 'validation'], shuffle_files=True,
                                   data_dir=data_dir)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_validation = dataset_validation.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.map(utils.rgb_to_y).cache()
    dataset_validation = dataset_validation.map(utils.rgb_to_y).cache()
    dataset_train = dataset_train.map(utils.patches).unbatch().shuffle(buffer_size=1_000)

    #PSNR metric to be monitored while training.
    def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    # atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore (needed for tf2.7?)

    #Select the model to train.
    with mirrored_strategy.scope():
        if FLAGS.model_name == 'SESR':
          if FLAGS.linear_block_type=='collapsed':
            LinearBlock_fn = model_utils.LinearBlock_c
          else:
            LinearBlock_fn = model_utils.LinearBlock_e
          model = sesr.SESR(
            m=FLAGS.m,
            feature_size=FLAGS.feature_size,
            LinearBlock_fn=LinearBlock_fn,
            quant_W=FLAGS.quant_W > 0,
            quant_A=FLAGS.quant_A > 0,
            gen_tflite = FLAGS.gen_tflite,
            mode='train')
    
        #Declare the optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, 
                                          amsgrad=True)
    
        #If scale == 4, base x2 model must be loaded for transfer learning.
        #Load the pretrained weights into the base model from x2 SISR:
        if FLAGS.scale == 4:
          base_model = tf.keras.models.load_model(PATH_2X, custom_objects={'psnr': psnr})
          layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
          for layer in model.layers:
            layer_name = layer.name
            if FLAGS.model_name == 'SESR':
              if layer_name != 'linear_block_{}'.format(FLAGS.m+1): #Last layer in x4 is not the same as that in x2 for SESR
                print(layer_name)
                layer.set_weights = layer_dict[layer_name].get_weights()

        #Compile and train the model.
        model.compile(optimizer=optimizer, loss='mae', metrics=[psnr])
    
    model.fit(dataset_train.batch(FLAGS.batch_size), 
              epochs=FLAGS.epochs, 
              validation_data=dataset_validation.batch(1), 
              validation_freq=1)
    model.summary()

    #Save the trained models.
    if FLAGS.model_name == 'SESR':
      final_save_path = BASE_SAVE_DIR+FLAGS.model_name+'_m{}_f{}_x{}_fs{}_{}Training_{}'.format(
                           FLAGS.m, FLAGS.int_features, FLAGS.scale, FLAGS.feature_size, 
                           FLAGS.linear_block_type, SUFFIX)
      model.save(final_save_path)
      model.save_weights(final_save_path + '/model_weights')

      #Get the TFLITE for custom image size
      if FLAGS.gen_tflite:
        y_lr = tf.random.uniform([1, FLAGS.tflite_height, FLAGS.tflite_width, 1], 
                                minval=0., maxval=1.)
        model_tflite = sesr.SESR(
          m=FLAGS.m,
          feature_size=FLAGS.feature_size,
          LinearBlock_fn=LinearBlock_fn,
          quant_W=FLAGS.quant_W > 0,
          quant_A=FLAGS.quant_A > 0,
          gen_tflite = FLAGS.gen_tflite,
          mode='infer')
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, 
                                      amsgrad=True)
        model_tflite.load_weights(final_save_path + '/model_weights')
        model_tflite.compile(optimizer=optimizer, loss='mae', metrics=[psnr])
        # build (execute forward pass)
        model_tflite(y_lr)
        utils.generate_int8_tflite(
          model_tflite,
          'model_quantized',
          final_save_path,
          fake_quant=True)


if __name__ == '__main__':
    tf.compat.v1.app.run()
