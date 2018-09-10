#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
train_filename=data/sejong.train.csv # Path of train dataset.
dev_filename=data/sejong.test.csv # Path of dev dataset.
out_dir=parser/model_iss57 # Store log/model files.
device=gpu # device to use
debug=false # use debug mode
num_train_epochs=100 # Num epochs to train.
batch_size=128  # Batch size.
word_embed_size=200  # The embedding dimension for the word's embedding.
pos_embed_size=100  # The embedding dimension for the tag's embedding.
word_embed_file=embeddings/words.pos.normalize.vec  # Use the pre-trained embedding. If not provided, use random values.
pos_embed_file=embeddings/words.tag.original.vec  # Use the pre-trained embedding. If not provided, use random values.
word_embed_matrix_file=embeddings/word_embed_matrix.txt
pos_embed_matrix_file=embeddings/pos_embed_matrix.txt
embed_dropout=0.33
############## END #####################

[ -d foo ] || mkdir ${out_dir}

export CUDA_VISIBLE_DEVICES=1
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "train"
python -m parser.parser \
    --train_filename=${train_filename} \
    --dev_filename=${dev_filename} \
    --word_embed_file=${word_embed_file} \
    --pos_embed_file=${pos_embed_file} \
    --num_train_epochs=${num_train_epochs} \
    --batch_size=${batch_size} \
    --word_embed_size=${word_embed_size} \
    --pos_embed_size=${pos_embed_size} \
    --embed_dropout=${embed_dropout} \
    --out_dir=${out_dir} \
    --device=${device} \
    --debug=${debug} \
    --word_embed_matrix_file=${word_embed_matrix_file} \
    --pos_embed_matrix_file=${pos_embed_matrix_file}