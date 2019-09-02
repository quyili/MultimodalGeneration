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


def selecter(
        INDEX_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/SELECT_FIG/INDEX/",
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/F_and_M_1650/F/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/F_and_M_1650/M/",
        SRC_PATH_3="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T1/",
        SRC_PATH_4="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T2/",
        SRC_PATH_5="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T1c/",
        SRC_PATH_6="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/Flair/",
        SRC_PATH_7="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/LabelV/",
        SRC_PATH_8="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/seg_res/selected/Label_Fake/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/SELECT_FIG/",
):
    try:
        os.makedirs(SAVE_PATH + "F")
        os.makedirs(SAVE_PATH + "M")
        os.makedirs(SAVE_PATH + "T1")
        os.makedirs(SAVE_PATH + "T2")
        os.makedirs(SAVE_PATH + "T1c")
        os.makedirs(SAVE_PATH + "Flair")
        os.makedirs(SAVE_PATH + "Label_T")
        os.makedirs(SAVE_PATH + "Label_F")
    except os.error:
        pass

    index_files = read_filename(INDEX_PATH, shuffle=False)
    for i in range(len(index_files)):
        os.system("cp " + SRC_PATH_1 + index_files[i] + " " + SAVE_PATH + "F/" + index_files[i])
        os.system("cp " + SRC_PATH_2 + index_files[i] + " " + SAVE_PATH + "M/" + index_files[i])
        os.system("cp " + SRC_PATH_3 + index_files[i] + " " + SAVE_PATH + "T1/" + index_files[i])
        os.system("cp " + SRC_PATH_4 + index_files[i] + " " + SAVE_PATH + "T2/" + index_files[i])
        os.system("cp " + SRC_PATH_5 + index_files[i] + " " + SAVE_PATH + "T1c/" + index_files[i])
        os.system("cp " + SRC_PATH_6 + index_files[i] + " " + SAVE_PATH + "Flair/" + index_files[i])
        os.system("cp " + SRC_PATH_7 + index_files[i] + " " + SAVE_PATH + "Label_T/" + index_files[i])
        os.system("cp " + SRC_PATH_8 + index_files[i] + " " + SAVE_PATH + "Label_F/" + index_files[i])


if __name__ == '__main__':
    selecter(
        INDEX_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/SELECT_FIG/INDEX/",
        SRC_PATH_1="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/F_and_M_1650/F/",
        SRC_PATH_2="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/F_and_M_1650/M/",
        SRC_PATH_3="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T1/",
        SRC_PATH_4="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T2/",
        SRC_PATH_5="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/T1c/",
        SRC_PATH_6="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/Flair/",
        SRC_PATH_7="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/test_images/LabelV/",
        SRC_PATH_8="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/G_MRI/new_f_to_mm_by_cGAN_4seg_1/seg_res/selected/Label_Fake/",
        SAVE_PATH="/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/SELECT_FIG/",
    )
