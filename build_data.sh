#!/bin/bash

#### Set your arguments here ####
############## START ###################
train_input_file=raw/sejong.ppos2.train.utf8.txt
train_output_file=raw/sejong.train.csv
dev_input_file=raw/sejong.ppos2.test.utf8.txt
dev_output_file=raw/sejong.test.csv
############## END #####################

#echo "building train_data"
#python -m parser.build_data \
#    --input_file=${train_input_file} \
#    --output_file=${train_output_file}

echo "building dev_data"
python -m parser.build_data \
    --input_file=${dev_input_file} \
    --output_file=${dev_output_file}