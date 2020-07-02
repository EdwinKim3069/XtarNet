import os

os.system('./run.sh 0 python run_exp.py '
          '--config configs/pretrain/mini-imagenet-resnet-snail.prototxt '
          # '--config configs/pretrain/tiered-imagenet-resnet-18.prototxt '
          '--dataset mini-imagenet '
          # '--dataset tiered-imagenet '
          '--data_folder DATA_ROOT/mini-imagenet/ '
          # '--data_folder DATA_ROOT/tiered-imagenet/ '
          '--results PRETRAIN_RESULTS '
          '--tag EXP_TAG '
          )