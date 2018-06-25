#-*- coding: utf-8 -*-
import os
import numpy as np
import config
import librosa
from librosa.display import specshow

#from PIL import Image
import matplotlib.pyplot as plt

def save_audio_spectrum_chroma(input_dir, file, output_dir):
    input_file_path = os.path.join(input_dir, file)
    x, fs = librosa.load(input_file_path, sr=config.SAMPLEING_RATE)
    S = librosa.stft(x, config.N_FFT)
    mag, phase = librosa.magphase(S)
    
    S = np.log1p(np.abs(mag[:config.HEIGHT, :config.WIDTH]))
    S_new = np.zeros((config.HEIGHT, config.WIDTH))
    S_new[:S.shape[0], :S.shape[1]] = S
    S = S_new

    fig, ax = plt.subplots()
    specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='linear', cmap='gray_r')
    #ax.axis('off')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    path = os.path.join(output_dir, os.path.splitext(file)[0]+'.png')
    fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi='figure')
    plt.close('all')


if __name__ == "__main__":
    
    print("Make spectrogram ...")
    folder = "./test/test_spectro_A"
    files = [f for f in os.listdir(os.path.join(folder))]
    for i in range(len(files)):
        save_audio_spectrum_chroma(folder, files[i], "./png/A_test_spectrogram_png")

    folder = "./test/test_spectro_B"
    files = [f for f in os.listdir(os.path.join(folder))]
    for i in range(len(files)):
        save_audio_spectrum_chroma(folder, files[i], "./png/B_test_spectrogram_png")

    folder = "./result/result_inference_mp3_output_is_A"
    files = [f for f in os.listdir(os.path.join(folder))]
    for i in range(len(files)):
        save_audio_spectrum_chroma(folder, files[i], "./png/B_to_A_spectrogram_png")

    folder = "./result/result_inference_mp3_output_is_B"
    files = [f for f in os.listdir(os.path.join(folder))]
    for i in range(len(files)):
        save_audio_spectrum_chroma(folder, files[i], "./png/A_to_B_spectrogram_png")
    

    print("Total spectrogram ...")
    folder="./source/"
    save_audio_spectrum_chroma(folder, "a_test.mp3", "./png/")

    folder="./source/"
    save_audio_spectrum_chroma(folder, "b_test.mp3", "./png/")

    folder="./result/"
    save_audio_spectrum_chroma(folder, "result_a_to_b.wav", "./png/")

    folder="./result/"
    save_audio_spectrum_chroma(folder, "result_b_to_a.wav", "./png/")
