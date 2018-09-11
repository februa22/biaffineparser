# -*- coding: utf-8 -*-
""" convert sejong_dataset to train,dev,test file_format """
import pandas as pd
import re
import argparse
import csv


def add_arguments(parser):
    # data
    parser.add_argument('--input_file', type=str,
                        help='Path of input file')
    parser.add_argument('--output_file', type=str,
                        help='Path of output file for training, validating, testing')


def main(flags):
    input_file = flags.input_file
    output_file = flags.output_file

    try:
        print(f'reading input_file={input_file}')
        column_names = ['eoj_id', 'head_id', 'label', 'eoj', 'eoj_kangwon']
        sejong_df = pd.read_csv(input_file, sep='\t', skip_blank_lines=True,
                                comment=';', names=column_names, quoting=csv.QUOTE_NONE)
    except Exception as e:
        print(
            f'error occured while reading input_file: input_file={input_file}, error={e}')
        print('quit')
        exit()

    print(f"writing START: output_file={output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # write header
        f.write('\t'.join(['sent_id', 'eoj_id', 'eoj',
                           'pos', 'head_id', 'label', 'char']) + "\n")
        sent_id = 0
        for index, row in sejong_df.iterrows():
            # initiailization
            eoj_id = row['eoj_id']
            if eoj_id == 1:
                sent_id += 1
            eoj = row['eoj_kangwon']
            #eoj_only = '|'.join([morph[:morph.rfind('/')] for morph in str(eoj).strip().split('|')])
            char = '|'.join(['|'.join(morph[:morph.rfind('/')])
                             for morph in str(eoj).strip().split('|')])

            head_id = row['head_id']
            label = row['label']
            pos = '|'.join([morph[morph.rfind('/')+1:]
                            for morph in str(eoj).strip().split('|')])

            f.write('\t'.join(
                [str(s) for s in [sent_id, eoj_id, eoj, pos, head_id, label, char]]) + "\n")
            if index % 10000 == 0:
                print(f'writing line index={index}')
    print('writing output_file END')
    return


if __name__ == '__main__':

    print('build data START')
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    main(FLAGS)
    print('build data END')
