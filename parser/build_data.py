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
    parser.add_argument('--delimiter', type=str, default='|',
                        help='delimter to split eoj to tokens')
    parser.add_argument('--mode', type=str, default='w',
                        help='mode to write or append (w | a)')
    parser.add_argument('--eoj_index', type=int, default=4,
                        help='index of eoj column in raw csv file')


def main(flags):
    input_file_path = flags.input_file
    output_file_path = flags.output_file
    sent_id = 0

    #if appending, get last sentnece_id
    if flags.mode == 'a':
        with open(output_file_path, 'r', encoding='utf-8') as output_file:
            for index, line in enumerate(output_file):
                if not line.strip() or line.strip().startswith(';'):
                    continue
                line = line.strip()
                sent_id = line.split('\t')[0]
            print(f'appending mode: start sent_id={sent_id}')
            print(f'appending mode: last_line={line}')
            sent_id = int(sent_id)

    #reading and writing start
    print(f'reading and writing START: input_file={input_file_path}, output={output_file_path}')
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, flags.mode, encoding='utf-8') as output_file:
            if flags.mode == 'w':
                output_file.write('sent_id\teoj_id\teoj\tpos\thead_id\tlabel\tchar\n')
            for index, line in enumerate(input_file):
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                row = line.split('\t')
                eoj_id = row[0]
                if int(eoj_id) == 1:
                    sent_id += 1
                eoj = row[flags.eoj_index]
                char = '|'.join(['|'.join(morph[:morph.rfind('/')])
                                 for morph in str(eoj).strip().split(flags.delimiter)])
                head_id = row[1]
                label = row[2]
                pos = '|'.join([morph[morph.rfind('/')+1:]
                                for morph in str(eoj).strip().split(flags.delimiter)])
                output_file.write('\t'.join(
                    [str(s) for s in [sent_id, eoj_id, eoj, pos, head_id, label, char]]) + "\n")
                if index % 10000 == 0:
                    print(f'writing line index={index}')
        print('reading and writing END')
        return


if __name__ == '__main__':

    print('build data START')
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    main(FLAGS)
    print('build data END')