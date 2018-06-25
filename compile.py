#-*- coding: utf-8 -*-
import os
from pydub import AudioSegment
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--istrain', action='store_true', dest='istrain', default=False)
parser.add_argument('--istest', action='store_true', dest='istest', default=False)
parser.add_argument('--isdelete', action='store_true', dest='isdelete', default=False)
parser.add_argument('--mkresult', action='store_true', dest='mkresult', default=False)
FLAGS = parser.parse_args()

PROJECT_FOLDER_NAME= "gan"
OVERWATCH_DATASET_PATH = "D:\\18_01_MLA\\dataset\\오버워치 음성(KR)"
A_NAME = "솔저76"
B_NAME = "파라"


def compile_all_audios(input_dir, input_list, output_path, type='mp3'):
    if not FLAGS.mkresult:
        complied_audio = None
        for idx in range(len(input_list)):
            if (idx == 0):
                compiled_audio = AudioSegment.from_mp3(os.path.join(input_dir, input_list[idx]))
            else:
                temp_audio = AudioSegment.from_mp3(os.path.join(input_dir, input_list[idx]))
                compiled_audio = compiled_audio + temp_audio
        try:
            if(type == 'mp3'):
                compiled_audio.export(output_path, format='mp3')
            elif(type == 'wav'):
                compiled_audio.export(output_path, format='wav')
        except UnboundLocalError as e:
            pass
    else:
        complied_audio = None
        for idx in range(len(input_list)):
            if (idx == 0):
                compiled_audio = AudioSegment.from_wav(os.path.join(input_dir, input_list[idx]))
            else:
                temp_audio = AudioSegment.from_wav(os.path.join(input_dir, input_list[idx]))
                compiled_audio = compiled_audio + temp_audio
        try:
            if(type == 'mp3'):
                compiled_audio.export(output_path, format='mp3')
            elif(type == 'wav'):
                compiled_audio.export(output_path, format='wav')
        except UnboundLocalError as e:
            pass


def collect_larget_than_one_sec(input_dir, input_list, output_path):
    BASE_SEC = 0
    for idx in range(len(input_list)):
        temp_audio = AudioSegment.from_ogg(os.path.join(input_dir, input_list[idx]))
        if (len(temp_audio) / 1000.0) > BASE_SEC:
            if os.path.exists(os.path.join(output_path, input_list[idx])):
                pass
            else:
                temp_audio.export(os.path.join(output_path, os.path.splitext(input_list[idx])[0] + ".mp3"), format="mp3")


def read_csv_and_delete_specific_audio_file(path):
    remove_txt_keywords=['신음', '웃음', '기계음', '기합', '추위', '한숨', '기침']
    # id, filename, hero, txt, info
    csv_file = pd.read_csv(path, engine='python', encoding='utf-8')
    cnt = 0;
    for index, row in csv_file.iterrows():
        if row['txt'] in remove_txt_keywords:
            try:
                print(os.path.join(OVERWATCH_DATASET_PATH, row['hero'], "Sound Dump", row['filename']+'.ogg'))
                os.remove(os.path.join(OVERWATCH_DATASET_PATH, row['hero'], "Sound Dump", row['filename']+'.ogg'))
                cnt += 1
            except FileNotFoundError as e:
                pass
    print("Total remove : ", cnt)


if __name__ == "__main__":
    if FLAGS.isdelete:
        read_csv_and_delete_specific_audio_file(path=os.path.join(OVERWATCH_DATASET_PATH, "overwatch.csv"))

    if FLAGS.istrain:
        # Preprocess : larger than 1 sec
        print("Move Train set ...")
        source_a = "D:\\18_01_MLA\\dataset\\오버워치 음성(KR)\\%s\\Sound Dump" % A_NAME
        source_b = "D:\\18_01_MLA\\dataset\\오버워치 음성(KR)\\%s\\Sound Dump" % B_NAME
        a_list = [f for f in os.listdir(source_a + '/')]
        b_list = [f for f in os.listdir(source_b + '/')]
        collect_larget_than_one_sec(source_a, a_list, "./source/inputs_a/")
        collect_larget_than_one_sec(source_b, b_list, "./source/inputs_b/")

        # Make Train set
        print("Preprocess Train set ...")
        source_a = "C:\\Users\\jwlim\\PycharmProjects\\%s\\source\\inputs_a"%PROJECT_FOLDER_NAME
        source_b = "C:\\Users\\jwlim\\PycharmProjects\\%s\\source\\inputs_b"%PROJECT_FOLDER_NAME
        a_list = [f for f in os.listdir(source_a + '/')]
        b_list = [f for f in os.listdir(source_b + '/')]
        compile_all_audios(source_a, a_list, "./source/a.mp3", type='mp3')
        compile_all_audios(source_b, b_list, "./source/b.mp3", type='mp3')

    if FLAGS.istest:
        # Preprocess : larger than 1 sec
        print("Move Test set ...")
        source_a = "D:\\18_01_MLA\\dataset\\오버워치 음성(KR)\\%s\\test" % A_NAME
        source_b = "D:\\18_01_MLA\\dataset\\오버워치 음성(KR)\\%s\\test" % B_NAME
        a_list = [f for f in os.listdir(source_a + '/')]
        b_list = [f for f in os.listdir(source_b + '/')]
        collect_larget_than_one_sec(source_a, a_list, "./source/inputs_a_test/")
        collect_larget_than_one_sec(source_b, b_list, "./source/inputs_b_test/")

        # Make Test set
        print("Preprocess Test set ...")
        source_a = "C:\\Users\\jwlim\\PycharmProjects\\%s\\source\\inputs_a_test"%PROJECT_FOLDER_NAME
        source_b = "C:\\Users\\jwlim\\PycharmProjects\\%s\\source\\inputs_b_test"%PROJECT_FOLDER_NAME
        a_list = [f for f in os.listdir(source_a + '/')]
        b_list = [f for f in os.listdir(source_b + '/')]
        compile_all_audios(source_a, a_list, "./source/a_test.mp3", type='mp3')
        compile_all_audios(source_b, b_list, "./source/b_test.mp3", type='mp3')

    if FLAGS.mkresult:
        source_a = "./result/result_inference_mp3_output_is_A"
        source_b = "./result/result_inference_mp3_output_is_B"
        a_list = [f for f in os.listdir(source_a + '/')]
        b_list = [f for f in os.listdir(source_b + '/')]
        compile_all_audios(source_a, a_list, "./result/result_b_to_a.wav", type='wav')
        compile_all_audios(source_b, b_list, "./result/result_a_to_b.wav", type='wav')

