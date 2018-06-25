# -*- coding: utf-8 -*-
import os
import subprocess

Preprocessed_A_npy = "./test/test_preprocessed_A_npy"
Preprocessed_B_npy = "./test/test_preprocessed_B_npy"
checkpoint = 'model-60000'

files_01 = [f for f in os.listdir(Preprocessed_A_npy + '/')]
files_02 = [f for f in os.listdir(Preprocessed_B_npy + '/')]
files_01.sort()
files_02.sort()

len_f01 = len(files_01)
len_f02 = len(files_02)

try:
    for i in range(len_f01):
        subprocess.call("python model.py --istarget --gen_B --input_dir=%s --input_data=%s --checkpoint=%s"%(Preprocessed_A_npy, files_01[i], checkpoint), shell=True)

    for i in range(len_f02):
        subprocess.call("python model.py --istarget --gen_A --input_dir=%s --input_data=%s --checkpoint=%s"%(Preprocessed_B_npy, files_02[i], checkpoint), shell=True)
except KeyboardInterrupt as e:
    exit()
