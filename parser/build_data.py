# -*- coding: utf-8 -*-
""" convert sejong_dataset to train,dev,test file_format """
import pandas as pd
import re
import argparse

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
        column_names = ['word_id', 'head_id', 'dependancy_label', 'word_and_morph', 'word_and_morph_kangwon']
        sejong_df = pd.read_csv(input_file, sep='\t', skip_blank_lines=True, comment=';', names=column_names)
    except Exception as e:
        print(f'error occured while reading input_file: input_file={input_file}, error={e}')
        print('quit')
        exit()
    
    print(f"writing START: output_file={output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        #write headr line
        #f.write('s,i,w,l,x,p,g,f,e,Type,Tense,Person,Foreign,Voice,Reflex,Definite,type,Mood,Number,Case,Degree,Poss,Gender\n')
        f.write('s\ti\tw\tl\tx\tp\tg\tf\te\tType\tTense\tPerson\tForeign\tVoice\tReflex\tDefinite\ttype\tMood\tNumber\tCase\tDegree\tPoss\tGender\n')
        sentence_index = 0
        for index, row in sejong_df.iterrows():
            #initiailization
            _s = '_'
            _i = '_'
            _w = '_'
            _l = '_'
            _x = '_'
            _p = '_'
            _g = '_'
            _f = '_'
            _e = '_'
            _Type = '_'
            _Tense = '_'
            _Person = '_'
            _Foreign = '_'
            _Voice = '_'
            _Reflex = '_'
            _Definite = '_'
            _type = '_'
            _Mood = '_'
            _Number = '_'
            _Case = '_'
            _Degree = '_'
            _Poss = '_'
            _Gender = '_'
            
            _i = int(row['word_id'])
            _g = row['head_id'] 
            _f = row['dependancy_label'] if _g != 0 else 'root'
            _x  = ''.join(re.findall('(\[.*?\]|\|)', row['word_and_morph']))
            _p = _x
            _w = re.sub('\[.*?\]', '', row['word_and_morph'])
            _l = _w
            if _i == 1:
                sentence_index += 1
            _s = sentence_index
            #f.write(f'{_s},{_i},{_w},{_l},{_x},{_p},{_g},{_f},{_e},{_Type},{_Tense},{_Person},{_Foreign},{_Voice},{_Reflex},{_Definite},{_type},{_Mood},{_Number},{_Case},{_Degree},{_Poss},{_Gender}\n')
            f.write(f'{_s}\t{_i}\t{_w}\t{_l}\t{_x}\t{_p}\t{_g}\t{_f}\t{_e}\t{_Type}\t{_Tense}\t{_Person}\t{_Foreign}\t{_Voice}\t{_Reflex}\t{_Definite}\t{_type}\t{_Mood}\t{_Number}\t{_Case}\t{_Degree}\t{_Poss}\t{_Gender}\n')
            if index % 10000 == 0:
                print(f'writing line index={index}')
    print('writing output_file END')
    return
    
if __name__ == '__main__':
    print('building data START')
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    main(FLAGS)
    print('building data END')