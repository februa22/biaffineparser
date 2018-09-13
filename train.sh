#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=data/sejong.char.train.csv # Path of train dataset.
dev_filename=data/sejong.char.test.csv # Path of dev dataset.
out_dir=parser/model_iss72-2 # Store log/model files.
word_embed_file=embeddings/words.pos.original.vec  # Use the pre-trained embedding. If not provided, use random values.
pos_embed_file=embeddings/words.tag.original.vec  # Use the pre-trained embedding. If not provided, use random values.
char_embed_file= #new
word_embed_size=200  # The embedding dimension for the word's embedding.
pos_embed_size=100  # The embedding dimension for the tag's embedding.
char_embed_size=200
embed_dropout=0.33
num_train_epochs=50 # Num epochs to train.
batch_size=128  # Batch size.
inference_input_file=data/sejong.char.train.csv
inference_output_file=${out_dir}/sejong.char.train.inference.tsv
device=gpu # device to use
debug=false # use debug mode
word_embed_matrix_file=embeddings/word_embed_matrix.txt
pos_embed_matrix_file=embeddings/pos_embed_matrix.txt
char_embed_matrix_file=embeddings/char_embed_matrix.txt #new
num_lstm_layers=4
############## END #####################

[ -d foo ] || mkdir ${out_dir}

export CUDA_VISIBLE_DEVICES=1
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename} \
    --out_dir=${out_dir} \
    --word_embed_file=${word_embed_file} \
    --pos_embed_file=${pos_embed_file} \
    --char_embed_file=${char_embed_file} \
    --word_embed_size=${word_embed_size}    \
    --char_embed_size=${char_embed_size}  \
    --embed_dropout=${embed_dropout} \
    --num_train_epochs=${num_train_epochs} \
    --batch_size=${batch_size} \
    --inference_input_file=${inference_input_file} \
    --inference_output_file=${inference_output_file} \
    --device=${device} \
    --debug=${debug} \
    --word_embed_matrix_file=${word_embed_matrix_file} \
    --pos_embed_matrix_file=${pos_embed_matrix_file} \
    --char_embed_matrix_file=${char_embed_matrix_file} \
    --num_lstm_layers=${num_lstm_layers}
