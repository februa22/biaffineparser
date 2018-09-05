#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=data/sejong.train.csv # Path of train dataset.
dev_filename=data/sejong.test.csv # Path of dev dataset.
out_dir=parser/model_log # Store log/model files.
device=gpu # device to use
debug=false # use debug mode
num_train_epochs=100 # Num epochs to train.
batch_size=2  # Batch size.
############## END #####################

[ -d foo ] || mkdir ${out_dir}

echo "train"
python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename}  \
    --out_dir=${out_dir}  \
    --device=${device} \
    --debug=${debug} \
    --num_train_epochs=${num_train_epochs} \
    --batch_size=${batch_size}
