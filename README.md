## SUPER-EFFICIENT SUPER RESOLUTION (SESR)

Code to accompany the paper: Collapsible Linear Blocks for Super-Efficient Super Resolution (https://arxiv.org/abs/2103.09404).

With similar or better image quality, SESR achieves 2x to 330x improvement (x2 and x4 super resolution) in Multiply-Accumulate (MAC) operations compared to existing methods. 

![SESR Achieves State-of-the-art Super Resolution Results](/SESR_results.png)

## Latest Updates
**[New]** Quantization-Aware-Training support added for SESR networks. See "Running Quantization-Aware Training (QAT) and generating a TFLITE file" section below. Full int8 quantization (i.e., both weights and activations are quantized to 8-bits) of SESR results in minimal loss of PSNR: FP32 trained SESR-M5 achieves about 35.20dB PSNR on DIV2K dataset. INT8 SESR-M5 achieves 35.00dB PSNR.

**[New]** TFLITE (Int8) can also be generated after QAT. See "Running Quantization-Aware Training (QAT) and generating a TFLITE file" section below.


## Prerequisites
It is recommended to use a conda environment with python 3.6. Start by installing the requirements:
Minimum requirements: tensorflow-gpu>=2.2 and tensorflow_datasets>=4.1. Install these using the following command:

`./install_requirements.sh`

PLEASE USE TENSORFLOW-GPU VERSION>=2.3 FOR QAT AND TFLITE SUPPORT.


## New Efficient Training Methodology
The training time would increase if we directly train collapsible linear blocks in the expanded space and collapse them later. To address this, we developed an efficient implementation of SESR: We collapse the "Linear Blocks" at each training step (using Algorithms 1 and 2 shown in the paper), and then use this collapsed weight to perform forward pass convolutions. Since model weights are very small tensors compared to feature maps, this collapsing takes a very small time. _The training (backward pass) still updates the weights in the expanded space but the forward pass happens in collapsed space even during training_ (see figure below). Therefore, training the collapsible linear blocks is very efficient.

For the SESR-M5 network and a batch of 32 [64x64] images, training in expanded space takes 41.77B MACs for a single forward pass, whereas our efficient implementation takes only 1.84B MACs. Similar improvements happen in GPU memory and backward pass (due to reduced size of layerwise Jacobians). 

![Expanded Training vs. Collapsed Training](/collapsed_training.png)

## Training x2 SISR:

Train SESR-M5 network with m = 5, f = 16, feature_size = 256, with collapsed linear block:

`python train.py`

Train SESR-M5 network with m = 5, f = 16, feature_size = 256, with expanded linear block:

`python train.py --linear_block_type expanded`

Train SESR-M11 network with m = 11, f = 16, feature_size = 64, with collapsed linear block:

`python train.py --m 11 --feature_size 64`

Train SESR-XL network with m = 11, f = 16, feature_size = 64, with collapsed linear block:

`python train.py --m 11 --int_features 32 --feature_size 64`


## Training x4 SISR: Requires a corresponding pretrained x2 model to be present in the directory logs/x2_models/

Train SESR-M5 network with m = 5, f = 16, feature_size = 256, with collapsed linear block:

`python train.py --scale 4`

Train SESR-M5 network with m = 5, f = 16, feature_size = 256, with expanded linear block:

`python train.py --linear_block_type expanded --scale 4`

Train SESR-M11 network with m = 11, f = 16, feature_size = 64, with collapsed linear block:

`python train.py --m 11 --feature_size 64 --scale 4`

Train SESR-XL network with m = 11, f = 16, feature_size = 64, with collapsed linear block:

`python train.py --m 11 --int_features 32 --feature_size 64 --scale 4`

## Running Quantization-Aware Training (QAT) and generating a TFLITE file

Run the following command to quantize the network while training and for generating a TFLITE (for x2 SISR, SESR-M5 network):

`python train.py --quant_W --quant_A --gen_tflite`

By default, the generated TFLITE inputs 1080p (1920x1080) image and outputs an upscaled image (based on x2 or x4 scale).


## File description
| File | Description |
| ------ | ------ |
| train.py | Contains main training and eval loop for DIV2K dataset |
| utils.py | Dataset utils and preprocessing |
| models/sesr.py | Contains main SESR network class |
| models/model_utils.py| Contains the expanded and collapsed linear blocks (to be used inside SESR network) |

## Flag description and location:
| Flag | Filename | Description | Default value |
| ------ | ------ | ------ | ------ |
| epochs | train.py | Number of epochs to train | 300 |
| batch_size | train.py | Batch size during training | 32 |
| learning_rate | train.py | Learning rate for ADAM | 2e-4 |
| model_name | train.py | Name of the model | 'SESR' |
| quant_W | train.py | Quantize Weights (8-bits) | False |
| quant_A | train.py | Quantize Activations (8-bits) | False |
| gen_tflite | train.py | Generate int8 TFLITE after quantization-aware training is complete | False |
| tflite_height | train.py | Height of Low-Resolution image in TFLITE | 1080 |
| tflite_width | train.py | Width of Low-Resolution image in TFLITE | 1920 |
| scale | utils.py | Scale of SISR (either x2 or x4 SISR) | 2 |
| feature_size | models/sesr.py | Number of features inside linear blocks (used for SESR only) | 256 |
| int_features | models/sesr.py | Number of intermediate features within SESR (parameter f in paper). Used for SESR. | 16 |
| m | models/sesr.py | Number of 3x3 layers (parameter m in paper). Used for SESR. | 5 |
| linear_block_type | models/sesr.py | Specify whether to train a linear block which does an online collapsing during training, or a full expanded linear block: Options: "collapsed" [DEFAULT] or "expanded" | 'collapsed' |

