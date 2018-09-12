#!/bin/bash

#### Set your arguments here ####
############## START ###################
train_input_file=raw/sejong.ppos2.train.utf8.error.fixed.txt
train_output_file=data/sejong.train.compare.csv
dev_input_file=raw/sejong.ppos2.test.utf8.error.fixed.txt
dev_output_file=data/sejong.test.compare.csv
############## END #####################

echo "building dev_data"
python -m parser.build_data \
    --input_file=${dev_input_file} \
    --output_file=${dev_output_file}
echo "building dev_data DONE"    

echo "building train_data"
python -m parser.build_data \
    --input_file=${train_input_file} \
    --output_file=${train_output_file}
echo "building train_data DONE"