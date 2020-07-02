import os

gpu_num = 0
nshot = 1
tag = 'XtarNet_miniImageNet_{}shot'.format(nshot)
# tag = 'XtarNet_tieredImageNet_{}shot'.format(nshot)

if 'mini' in tag:

    os.system(''
              './run.sh {} '
              'python run_exp.py '
              '--config configs/XtarNet/XtarNet-mini-imagenet-resnet-snail.prototxt '
              '--dataset mini-imagenet '
              '--data_folder DATA_ROOT/mini-imagenet/ '
              '--pretrain PRETRAIN_FOLDER/mini-imagenet/PRETRAINED_BACKBONE '
              '--nshot {} '
              '--nclasses_b 5 '
              '--results METATRAIN_RESULTS '
              # '--eval '
              # '--retest '
              '--tag {} '.format(gpu_num, nshot, tag)
              )

if 'tiered' in tag:

    os.system(''
              './run.sh {} '
              'python run_exp.py '
              '--config configs/XtarNet/XtarNet-tiered-imagenet-resnet-18.prototxt '
              '--dataset tiered-imagenet '
              '--data_folder DATA_ROOT/tiered-imagenet/ '
              '--pretrain PRETRAIN_FOLDER/tiered-imagenet/PRETRAINED_BACKBONE '
              '--nshot {} '
              '--nclasses_b 5 '
              '--results METATRAIN_RESULTS '
              # '--eval '
              # '--retest '
              '--tag {} '.format(gpu_num, nshot, tag)
              )