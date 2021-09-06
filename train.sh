# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash


TRAINDATA="${HOME}/DATASETS/MODIS_EURE4CA/train_d"
VALDATA="${HOME}/DATASETS/MODIS_EURE4CA/study_d"
PATHSAVE="${HOME}/MODELS/autoencoder.model"
LR=0.05
EPOCHS=100
PYTHON="python"

# with training the batch norm
# 72.0 mAP
$PYTHON train.py --train_dataset $TRAINDATA --val_dataset $VALDATA --path_save $PATHSAVE --lr $LR --epochs $EPOCHS