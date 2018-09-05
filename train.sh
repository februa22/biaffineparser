#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=data/train_conll17.csv # Path of train dataset.
dev_filename=data/dev_conll17.csv # Path of dev dataset.
out_dir=parser/model # Store log/model files.
device=gpu # device to use
debug=true # use debug mode
num_train_epochs=100 # Num epochs to train.
batch_size=2
############## END #####################

echo "train"
python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename}  \
    --out_dir=${out_dir}  \
    --device=${device} \
    --debug=${debug} \
    --num_train_epochs=${num_train_epochs} \
    --batch_size=${batch_size}
