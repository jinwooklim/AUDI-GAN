# -*- coding: utf-8 -*-
import os

folder_list = [
        "./source",
        "./source/inputs_a", 
        "./source/inputs_b",
        "./source/inputs_a_test",
        "./source/inputs_b_test",
        
        "./png",
        "./png/A_test_spectrogram_png",
        "./png/B_test_spectrogram_png",
        "./png/A_to_B_spectrogram_png",
        "./png/B_to_A_spectrogram_png",
        
        "./train",
        "./train/preprocessed_A_npy",
        "./train/preprocessed_B_npy",
        "./train/spectro_A",
        "./train/spectro_B",
        "./train/valid_npy",
        "./train/valid_mp3",
        
        "./test",
        "./test/test_preprocessed_A_npy",
        "./test/test_preprocessed_B_npy",
        "./test/test_spectro_A",
        "./test/test_spectro_B",
        
        "./result",
        "./result/result_inference_npy_output_is_A",
        "./result/result_inference_npy_output_is_B",
        "./result/result_inference_mp3_output_is_A",
        "./result/result_inference_mp3_output_is_B"]

for i in range(len(folder_list)):
    try:
        os.mkdir(folder_list[i])
        print("sucess : ", folder_list[i])
    except FileExistsError as e:
        pass
