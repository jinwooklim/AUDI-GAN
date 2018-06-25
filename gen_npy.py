#-*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import librosa
import config
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument('--iscut', action='store_true', dest='iscut', default=False)
parser.add_argument('--isnpy', action='store_true', dest='isnpy', default=False)
parser.add_argument('--input_dir', action='store', dest='input_dir', default="./")
parser.add_argument('--input_data', action='store', dest='input_data', default=False)
parser.add_argument('--output_dir', action='store', dest='output_dir', default="./")
FLAGS = parser.parse_args()

AUDIO_FOLDER_01 = './train/spectro_A'
AUDIO_FOLDER_02 = './train/spectro_B'
NPY_FOLDER_01 = './train/preprocessed_A_npy'
NPY_FOLDER_02 = './train/preprocessed_B_npy'

TEST_AUDIO_FOLDER_01 = './test/test_spectro_A'
TEST_AUDIO_FOLDER_02 = './test/test_spectro_B'
TEST_NPY_FOLDER_01 = './test/test_preprocessed_A_npy'
TEST_NPY_FOLDER_02 = './test/test_preprocessed_B_npy'

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
if not os.path.exists(AUDIO_FOLDER_01):
    os.mkdir(AUDIO_FOLDER_01)
if not os.path.exists(AUDIO_FOLDER_02):
    os.mkdir(AUDIO_FOLDER_02)
if not os.path.exists(NPY_FOLDER_01):
    os.mkdir(NPY_FOLDER_01)
if not os.path.exists(NPY_FOLDER_02):
    os.mkdir(NPY_FOLDER_02)
if not os.path.exists(TEST_AUDIO_FOLDER_01):
    os.mkdir(TEST_AUDIO_FOLDER_01)
if not os.path.exists(TEST_AUDIO_FOLDER_02):
    os.mkdir(TEST_AUDIO_FOLDER_02)
if not os.path.exists(TEST_NPY_FOLDER_01):
    os.mkdir(TEST_NPY_FOLDER_01)
if not os.path.exists(TEST_NPY_FOLDER_02):
    os.mkdir(TEST_NPY_FOLDER_02)


#the instructions in this function has been taken from DmitryUlyanov/neural-style-audio-torch
#link is https://github.com/DmitryUlyanov/neural-style-audio-torch/blob/master/get_spectrogram.py
def read_audio_spectum(file):
    x, fs = librosa.load(file, sr=config.SAMPLEING_RATE)
    s = librosa.stft(x, config.N_FFT) # return is stft matix
    mag, phase = librosa.magphase(s) # input is complex-valued spectogram, return are magnitude(d) and phase(phi)
    s = np.log1p(np.abs(mag[:config.HEIGHT,:config.WIDTH])) # add 1 elementwise and do log, log(1+x), 1 is a log_offset
    s_new = np.zeros((config.HEIGHT,config.WIDTH))
    s_new[:s.shape[0], :s.shape[1]] = s
    s = s_new
    return s, fs


def cut_mp3_and_save(input_dir, file, output_dir, dBFS, istest):
    AUDIO_WIDTH = 0.01 * config.WIDTH #1 # slice must int
    if not istest: # train
        AUDIO_OFFSET = (0.01*config.WIDTH)/2.0 #0.5 #1.5
    else: # test
        AUDIO_OFFSET = 0.01 * config.WIDTH
    
    startMin = 0
    startSec = 0

    endMin = 0
    endSec = AUDIO_WIDTH

    # Opening file and extracting segment
    song = AudioSegment.from_mp3(os.path.join(input_dir, file))
    print("%s's duration seconds"%file, song.duration_seconds)
    for idx in range(len(song)):
        if(np.rint(len(song)/(1000*AUDIO_OFFSET))-1 <= idx):
            break

        # Time to miliseconds
        startTime = startMin*60*1000 + startSec*1000
        endTime = endMin*60*1000 + endSec*1000
        
        # Slice
        extract = song[startTime:endTime]

        # Saving
        extract.export(os.path.join(output_dir, os.path.splitext(file)[0]+"_%05d.mp3"%idx), format="mp3", bitrate="192k")

        startSec += AUDIO_OFFSET
        endSec += AUDIO_OFFSET


def save_npy_array_by_spectrum(indir, outdir):
    files_list = [f for f in os.listdir(indir+'/')]
    files_list.sort()
    for file in files_list:
        a_audio, fs = read_audio_spectum(indir+'/'+file)
        a_audio = np.reshape(a_audio, newshape=(-1, config.WIDTH, 1))
        a_audio_npy = np.asarray(a_audio)
        np.save(os.path.join(outdir, os.path.splitext(file)[0]), a_audio_npy)


if __name__ == "__main__":
    if FLAGS.iscut:
        print("Full mp3 --> Splited mp3")
        a_sound = AudioSegment.from_mp3("./source/a.mp3")
        b_sound = AudioSegment.from_mp3("./source/b.mp3")
        cut_mp3_and_save("./source", "a.mp3", AUDIO_FOLDER_01, a_sound.dBFS, istest=False)
        cut_mp3_and_save("./source", "b.mp3", AUDIO_FOLDER_02, b_sound.dBFS, istest=False)

        a_sound = AudioSegment.from_mp3("./source/a_test.mp3")
        b_sound = AudioSegment.from_mp3("./source/b_test.mp3")
        cut_mp3_and_save("./source", "a_test.mp3", TEST_AUDIO_FOLDER_01, a_sound.dBFS, istest=True)
        cut_mp3_and_save("./source", "b_test.mp3", TEST_AUDIO_FOLDER_02, b_sound.dBFS, istest=True)

    elif FLAGS.isnpy:
        print("Splited mp3 --> npy")
        save_npy_array_by_spectrum(AUDIO_FOLDER_01, NPY_FOLDER_01)
        save_npy_array_by_spectrum(AUDIO_FOLDER_02, NPY_FOLDER_02)
        save_npy_array_by_spectrum(TEST_AUDIO_FOLDER_01, TEST_NPY_FOLDER_01)
        save_npy_array_by_spectrum(TEST_AUDIO_FOLDER_02, TEST_NPY_FOLDER_02)
