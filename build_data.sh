#!/bin/bash

#### Set your arguments here ####
############## START ###################
train_input_file=raw/sejong.ppos2.train.utf8.error.fixed.txt
train_output_file=data/sejong.char.train.csv
dev_input_file=raw/sejong.ppos2.test.utf8.error.fixed.txt
dev_output_file=data/sejong.char.test.csv
############## END #####################

#exit 0

echo "building dev_data"
python -m parser.build_data \
    --input_file=${dev_input_file} \
    --output_file=${dev_output_file}
echo "building dev_data DONE"    

#exit 0

echo "building train_data with sejong"
python -m parser.build_data \
    --input_file=${train_input_file} \
    --output_file=${train_output_file}
echo "building train_data DONE"

echo "building train_data with changwon and appending"
python -m parser.build_data \
    --input_file=raw/changwon_dep-2.txt \
    --output_file=${train_output_file} \
    --delimiter=+ \
    --mode=a \
    --eoj_index=3
echo "building train_data DONE"