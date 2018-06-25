# -*- coding: utf-8 -*-
import librosa
import os
import numpy as np
import argparse
import config

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument('--input_dir', action='store', dest='input_dir', default='result_npy')
parser.add_argument('--input', action="store", dest="input", default='')
parser.add_argument('--input_height', action="store", dest="input_height", default='')
parser.add_argument('--input_width', action="store", dest="input_width", default='')
parser.add_argument('--output_dir', action="store", dest="output_dir", default='result_mp3')
parser.add_argument('--output', action="store", dest="audio_output", default='')
FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

#this function has been taken from DmitryUlyanov/neural-style-audio-torch
#link is https://github.com/DmitryUlyanov/neural-style-audio-torch/blob/master/invert_spectrogram.py
def phase_restore(mag, random_phases, N):
    p = np.exp(1j * (random_phases))
    for i in range(N):
        _, p = librosa.magphase(librosa.stft(librosa.istft(mag * p), n_fft=config.N_FFT))
    return p


def gen_single_npy_to_audio():
    audio_files = np.load(os.path.join(FLAGS.input_dir, FLAGS.input))
    print("Processing ... ", FLAGS.input)
    gen_audio = audio_files
    gen_audio = np.reshape(gen_audio, newshape=(config.HEIGHT, config.WIDTH))

    req_audio = np.zeros([config.HEIGHT+1, gen_audio.shape[1]])
    req_audio[:gen_audio.shape[0]] = gen_audio

    random_phase = req_audio.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(req_audio) - 1), random_phase, N=100)
    y = librosa.istft((np.exp(req_audio) - 1) * p)
    librosa.output.write_wav(os.path.join(FLAGS.output_dir, os.path.splitext(FLAGS.input)[0]+".wav"), y, 22050, norm=False)


if __name__ == "__main__":
    try:
        gen_single_npy_to_audio()
    except librosa.util.exceptions.ParameterError:
        print("Audio buffer is not finite everywhere")
        exit()
