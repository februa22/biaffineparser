#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=raw/sejong.train.csv # Path of train dataset.
dev_filename=raw/sejong.test.csv # Path of dev dataset.
out_dir=parser/parser_model # Store log/model files.
device=gpu # device to use
debug=false # use debug mode
############## END #####################

echo "train"
python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename}  \
    --out_dir=${out_dir}  \
    --device=${device} \
    --debug=${debug}
