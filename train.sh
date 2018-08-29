#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=data/train_conll17.csv # Path of train dataset.
dev_filename=data/dev_conll17.csv # Path of dev dataset.
device=gpu # device to use
debug=false # use debug mode
############## END #####################

echo "train"
python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename}  \
    --device=${device} \
    --debug=${debug}
