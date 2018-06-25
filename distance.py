# -*- coding: utf-8 -*-
import os
import random
import numpy as np

random.seed(22)


def calculate_L2_distance(base_folder, compare_folder):
    train_files_path = os.listdir(base_folder)
    result_files_path = os.listdir(compare_folder)

    train_sample_path = random.sample(train_files_path, 30)
    test_sample_path = random.sample(result_files_path, 30)

    train_sample = np.asarray([np.load(os.path.join(base_folder, f)) for f in train_sample_path])
    test_sample = np.asarray([np.load(os.path.join(compare_folder, f)) for f in test_sample_path])

    L2_distance = np.linalg.norm(train_sample - test_sample)
    return L2_distance


if __name__ == "__main__":
    total_results = []
    results = []
    for idx in range(20):
        train_folder = "./train/preprocessed_A_npy"
        result_folder = "./test/test_preprocessed_A_npy"
        result = calculate_L2_distance(train_folder, result_folder)
        results.append(result)
    results = np.asarray(results)
    print("A_vs_testA : ")
    print(round(np.mean(results), 2), round(np.std(results), 2), end="\n\n")

    results = []
    for idx in range(20):
        train_folder = "./train/preprocessed_A_npy"
        result_folder = "./result/result_inference_npy_output_is_A"
        result = calculate_L2_distance(train_folder, result_folder)
        results.append(result)
    results = np.asarray(results)
    print("A_vs_genA : ")
    print(round(np.mean(results), 2), round(np.std(results), 2), end="\n\n")

    results = []
    for idx in range(20):
        train_folder = "./train/preprocessed_B_npy"
        result_folder = "./test/test_preprocessed_B_npy"
        result = calculate_L2_distance(train_folder, result_folder)
        results.append(result)
    results = np.asarray(results)
    print("B_vs_testB : ")
    print(round(np.mean(results), 2), round(np.std(results), 2), end="\n\n")

    results = []
    for idx in range(20):
        train_folder = "./train/preprocessed_B_npy"
        result_folder = "./result/result_inference_npy_output_is_B"
        result = calculate_L2_distance(train_folder, result_folder)
        results.append(result)
    results = np.asarray(results)
    print("B_vs_genB : ")
    print(round(np.mean(results), 2), round(np.std(results), 2), end="\n\n")
