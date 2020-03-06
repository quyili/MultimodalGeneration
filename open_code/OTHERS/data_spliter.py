# _*_ coding:utf-8 _*_
import os
import numpy as np


def read_filename(path, shuffle=False):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def split(
        SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        epoch=1,
        epoch_steps=15070):
    try:
        os.makedirs(SAVE_PATH + "T1")
        os.makedirs(SAVE_PATH + "T2")
        os.makedirs(SAVE_PATH + "T1c")
        os.makedirs(SAVE_PATH + "Flair")
        os.makedirs(SAVE_PATH + "Label")
    except os.error:
        pass
    l_val_files = read_filename(SRC_PATH + "Label")
    for i in range(int(epoch_steps * epoch)):
        os.system("cp " + SRC_PATH + "T1/" + l_val_files[i] + " " + SAVE_PATH + "T1/" + l_val_files[i])
        os.system("cp " + SRC_PATH + "T2/" + l_val_files[i] + " " + SAVE_PATH + "T2/" + l_val_files[i])
        os.system("cp " + SRC_PATH + "T1c/" + l_val_files[i] + " " + SAVE_PATH + "T1c/" + l_val_files[i])
        os.system("cp " + SRC_PATH + "Flair/" + l_val_files[i] + " " + SAVE_PATH + "Flair/" + l_val_files[i])
        os.system("cp " + SRC_PATH + "Label/" + l_val_files[i] + " " + SAVE_PATH + "Label/" + l_val_files[i])


if __name__ == '__main__':
    # split(
    #     SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
    #     SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
    #     epoch=1,
    #     epoch_steps=15070)
    # split(
    #     SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
    #     SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_2F_MRI/",
    #     epoch=2,
    #     epoch_steps=15070)
    # split(
    #     SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
    #     SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_3F_MRI/",
    #     epoch=3,
    #     epoch_steps=15070)

    split(
        SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/train",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_MRI/",
        epoch=1,
        epoch_steps=13420)
    split(
        SRC_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/train",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_MRI/",
        epoch=0.1,
        epoch_steps=13420)
