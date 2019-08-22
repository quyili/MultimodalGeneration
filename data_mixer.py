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

def mix(
        SRC_PATH_1= "/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
        SRC_PATH_2= "/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/selected_75350/",
        SAVE_PATH= "/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_50_synthetic_50_MRI/",
        epoch_1 = 0.5,
        epoch_2 = 0.5,
        epoch_steps = 15070):
    try:
        os.makedirs(SAVE_PATH + "T1")
        os.makedirs(SAVE_PATH + "T2")
        os.makedirs(SAVE_PATH + "T1c")
        os.makedirs(SAVE_PATH + "Flair")
        os.makedirs(SAVE_PATH + "Label")
    except os.error:
        pass

    l_val_files_1 = read_filename(SRC_PATH_1+"Label", shuffle=True)
    for i in range(int(epoch_steps*epoch_1)):
        os.system("cp " + SRC_PATH_1 + "trainT1/" + l_val_files_1[i] + " " + SAVE_PATH + "T1/real_" + l_val_files_1[i])
        os.system("cp " + SRC_PATH_1 + "trainT2/" + l_val_files_1[i] + " " + SAVE_PATH + "T2/real_" + l_val_files_1[i])
        os.system("cp " + SRC_PATH_1 + "trainT1c/" + l_val_files_1[i] + " " + SAVE_PATH + "T1c/real_" + l_val_files_1[i])
        os.system("cp " + SRC_PATH_1 + "trainFlair/" + l_val_files_1[i] + " " + SAVE_PATH + "Flair/real_" + l_val_files_1[i])
        os.system("cp " + SRC_PATH_1 + "trainLabel/" + l_val_files_1[i] + " " + SAVE_PATH + "Label/real_" + l_val_files_1[i])

    l_val_files_2 = read_filename(SRC_PATH_2 + "Label", shuffle=True)
    for i in range(int(epoch_steps * epoch_2)):
        os.system("cp " + SRC_PATH_2 + "T1/" + l_val_files_2[i] + " " + SAVE_PATH + "T1/notreal_" + l_val_files_2[i])
        os.system("cp " + SRC_PATH_2 + "T2/" + l_val_files_2[i] + " " + SAVE_PATH + "T2/notreal_" + l_val_files_2[i])
        os.system("cp " + SRC_PATH_2 + "T1c/" + l_val_files_2[i] + " " + SAVE_PATH + "T1c/notreal_" + l_val_files_2[i])
        os.system("cp " + SRC_PATH_2 + "Flair/" + l_val_files_2[i] + " " + SAVE_PATH + "Flair/notreal_" + l_val_files_2[i])
        os.system("cp " + SRC_PATH_2 + "Label/" + l_val_files_2[i] + " " + SAVE_PATH + "Label/notreal_" + l_val_files_2[i])

if __name__ == '__main__':
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/enhancement_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_enhancement_1F_MRI/",
        epoch_1=1,
        epoch_2=1,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/enhancement_2F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_enhancement_2F_MRI/",
        epoch_1=1,
        epoch_2=2,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/enhancement_3F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_enhancement_3F_MRI/",
        epoch_1=1,
        epoch_2=3,
        epoch_steps=15070)

    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_synthetic_1F_MRI/",
        epoch_1=1,
        epoch_2=1,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_2F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_synthetic_2F_MRI/",
        epoch_1=1,
        epoch_2=2,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_3F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_1F_synthetic_3F_MRI/",
        epoch_1=1,
        epoch_2=3,
        epoch_steps=15070)

    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_50_synthetic_50_MRI/",
        epoch_1=0.5,
        epoch_2=0.5,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_20_synthetic_80_MRI/",
        epoch_1=0.2,
        epoch_2=0.8,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_80_synthetic_20_MRI/",
        epoch_1=0.8,
        epoch_2=0.2,
        epoch_steps=15070)

    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_1F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_10_synthetic_1F_MRI/",
        epoch_1=0.1,
        epoch_2=1,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_2F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_10_synthetic_2F_MRI/",
        epoch_1=0.1,
        epoch_2=2,
        epoch_steps=15070)
    mix(
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/synthetic_3F_MRI/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/real_10_synthetic_3F_MRI/",
        epoch_1=0.1,
        epoch_2=3,
        epoch_steps=15070)
