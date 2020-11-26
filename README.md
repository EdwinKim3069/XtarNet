# XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning

This repository contains the code for the following ICML 2020 paper:

[**XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning**](https://arxiv.org/abs/2003.08561)

## Dependencies
This code was tested on the following environment:

* Ubuntu 16.04
* python 3.6
* cv2
* numpy
* pandas
* tensorflow==1.11
* tensorflow-gpu==1.9.0 (If cuda is available)
* tqdm

## Dataset

### miniImageNet
Download and decompress the file: [[miniImageNet](https://drive.google.com/file/d/197UFjJfkzXb89eFP0telPbib2tkt3ewF/view?usp=sharing)] 
(courtesy of [Mengye Ren](https://github.com/renmengye/inc-few-shot-attractor-public))
```
export DATA_ROOT={path/to/dataset}
mkdir -p $DATA_ROOT
cd $DATA_ROOT
mv ~/Downloads/mini-imagenet.tar .   # place "mini-imagenet.tar" in $DATA_ROOT
tar -xvf mini-imagenet.tar
rm -f mini-imagenet.tar
```

### tieredImageNet
Download and decompress the file: [[tieredImageNet](https://drive.google.com/file/d/1s6Pz5_YdLmcjpAFW5ciSpm_nNd3_YVmu/view?usp=sharing)] 
(courtesy of [Mengye Ren](https://github.com/renmengye/inc-few-shot-attractor-public))
```
export DATA_ROOT={path/to/dataset}
mkdir -p $DATA_ROOT
cd $DATA_ROOT
mv ~/Downloads/tiered-imagenet.tar .   # place "tiered-imagenet.tar" in $DATA_ROOT
tar -xvf tiered-imagenet.tar
rm -f tiered-imagenet.tar
```



## Usage


### Clone this repository
```
https://github.com/EdwinKim3069/XtarNet.git
cd XtarNet
```


### Pretraining

In order to pretrain backbone, run the python file ```run_pretrain.py```.

For miniImageNet experiment,
```python
import os
os.system('./run.sh 0 python run_exp.py '
          '--config configs/pretrain/mini-imagenet-resnet-snail.prototxt '
          '--dataset mini-imagenet '
          '--data_folder DATA_ROOT/mini-imagenet/ '
          '--results PATH/TO/SAVE_PRETRAIN_RESULTS '
          '--tag PATH/TO/EXP_TAG '
          )
```
For tieredImageNet experiment,
```python
import os
os.system('./run.sh 0 python run_exp.py '
          '--config configs/pretrain/tiered-imagenet-resnet-18.prototxt '
          '--dataset tiered-imagenet '
          '--data_folder DATA_ROOT/tiered-imagenet/ '
          '--results PATH/TO/SAVE_PRETRAIN_RESULTS '
          '--tag PATH/TO/EXP_TAG '
          )
```

### Meta-training

In order to run meta-training experiments, run the python file ```run_inc.py```.

For miniImageNet experiments,
```python
import os

nshot = 'NUMBER_OF_SHOTS'
tag = 'XtarNet_miniImageNet_{}shot'.format(nshot)
os.system('./run.sh 0 '
          'python run_exp.py '
          '--config configs/XtarNet/XtarNet-mini-imagenet-resnet-snail.prototxt '
          '--dataset mini-imagenet '
          '--data_folder DATA_ROOT/mini-imagenet/ '
          '--pretrain PATH/TO/PRETRAIN_RESULTS/TAG '
          '--nshot {} '
          '--nclasses_b 5 '
          '--results PATH/TO/SAVE_METATRAIN_RESULTS '
          # '--eval '
          # '--retest '
          '--tag {} '.format(nshot, tag)
          )
```

For tieredImageNet experiments,
```python
import os

nshot = 'NUMBER_OF_SHOTS'
tag = 'XtarNet_tieredImageNet_{}shot'.format(nshot)
os.system('./run.sh 0 '
          'python run_exp.py '
          '--config configs/XtarNet/XtarNet-tiered-imagenet-resnet-18.prototxt '
          '--dataset tiered-imagenet '
          '--data_folder DATA_ROOT/tiered-imagenet/ '
          '--pretrain PATH/TO/PRETRAIN_RESULTS/TAG '
          '--nshot {} '
          '--nclasses_b 5 '
          '--results PATH/TO/SAVE_METATRAIN_RESULTS '
          # '--eval '
          # '--retest '
          '--tag {} '.format(nshot, tag)
          )
```
If you want to evaluate the meta-trained model, add ```--eval``` and ```--retest``` flag for restoring a fully trained model and re-run eval.

## Acknowledgment
Our code is based on the implementations of [Incremental Few-Shot Learning with Attention Attractor Networks](https://github.com/renmengye/inc-few-shot-attractor-public).