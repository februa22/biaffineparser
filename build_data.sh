#!/bin/bash

#### Set your arguments here ####
############## START ###################
train_input_file=raw/sejong.ppos2.train.utf8.error.fixed.txt
train_output_file=data/sejong.char.train.csv
dev_input_file=raw/sejong.ppos2.test.utf8.error.fixed.txt
dev_output_file=data/sejong.char.test.csv
split_base=char #char | morph
############## END #####################

echo "building dev_data"
python -m parser.build_data \
    --input_file=${dev_input_file} \
    --output_file=${dev_output_file} \
    --split_base=${split_base}
echo "building dev_data DONE"    

echo "building train_data with sejong"
python -m parser.build_data \
    --input_file=${train_input_file} \
    --output_file=${train_output_file} \
    --split_base=${split_base}
echo "building train_data DONE"

#종료
exit 0

#창원대 데이처 추가시
echo "building train_data with changwon and appending"
python -m parser.build_data \
    --input_file=raw/changwon_dep-2.txt \
    --output_file=${train_output_file} \
    --delimiter=+ \
    --mode=a \
    --eoj_index=3  \
    --split_base=${split_base}
echo "building train_data DONE"