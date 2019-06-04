# _*_ coding:utf-8 _*_
import tensorflow as tf
from gen_model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK
import math


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return L_arr_


def read_files(x_path, l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    T1_img = SimpleITK.ReadImage(x_path + "/" + Label_train_files[index % train_range])
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    T1_arr_ = SimpleITK.GetArrayFromImage(T1_img)
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    T1_arr_ = T1_arr_.astype('float32')
    L_arr_ = L_arr_.astype('float32')
    return T1_arr_, L_arr_


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


Label_train_files = read_filename("data/BRATS2015/trainLabel")
train_L_arr_ = read_file("data/BRATS2015/trainLabel", Label_train_files, 0)
L0 = np.asarray(train_L_arr_ == 0., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
L1 = (train_L_arr_ * (train_L_arr_ == 1.)).reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
L2 = (train_L_arr_ * (train_L_arr_ == 2.)).reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
L3 = (train_L_arr_ * (train_L_arr_ == 3.)).reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
L4 = (train_L_arr_ * (train_L_arr_ == 4.)).reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
NL = np.concatenate([L0, L1, L2, L3, L4], axis=-1)
print(NL.shape)
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_L_arr_)), "samples/L_arr.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(NL[:, :, 0]), "samples/L0.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(NL[:, :, 1]), "samples/L1.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(NL[:, :, 2]), "samples/L2.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(NL[:, :, 3]), "samples/L3.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(NL[:, :, 4]), "samples/L4.tiff")
SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.sum(NL, axis=2)), "samples/NL_arr.tiff")
