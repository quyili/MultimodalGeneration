import numpy as np
import tensorflow as tf
import os
import SimpleITK

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('X', '/GPUFS/nsccgz_ywang_1/quyili/BRAST2015/train/T1', 'X files for training')
tf.flags.DEFINE_string('Y', '/GPUFS/nsccgz_ywang_1/quyili/BRAST2015/train/T1c', 'Y files for training')
tf.flags.DEFINE_string('Z', '/GPUFS/nsccgz_ywang_1/quyili/BRAST2015/train/T2', 'Z files for training')
tf.flags.DEFINE_string('W', '/GPUFS/nsccgz_ywang_1/quyili/BRAST2015/train/Flair', 'Z files for training')
tf.flags.DEFINE_string('L', '/GPUFS/nsccgz_ywang_1/quyili/BRAST2015/train/Label', 'Z files for training')
tf.flags.DEFINE_string('X_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainT1',
                       'X files for training')
tf.flags.DEFINE_string('Y_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainT1c',
                       'Y files for training')
tf.flags.DEFINE_string('Z_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainT2',
                       'Z files for training')
tf.flags.DEFINE_string('W_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainFlair',
                       'Z files for training')
tf.flags.DEFINE_string('L_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainLabel',
                       'Z files for training')
tf.flags.DEFINE_string('M_', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/data/BRATS2015/trainMask',
                       'Z files for training')

os.makedirs(FLAGS.X_)
os.makedirs(FLAGS.Y_)
os.makedirs(FLAGS.Z_)
os.makedirs(FLAGS.W_)
os.makedirs(FLAGS.L_)
os.makedirs(FLAGS.M_)


def read_img(path, train_files, index, norm=True):
    train_range = len(train_files)
    arr_ = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path + "/" + train_files[index % train_range])).astype(
        'float32')
    if norm:
        arr_ = (arr_ - np.min(arr_, axis=(0, 1, 2))) / (np.max(arr_, axis=(0, 1, 2)) - np.min(arr_, axis=(0, 1, 2)))
    return arr_[50:105, 38:222, 48:192]


def split_file(train_files, index):
    X = read_img(FLAGS.X, train_files, index)
    Y = read_img(FLAGS.Y, train_files, index)
    Z = read_img(FLAGS.Z, train_files, index)
    W = read_img(FLAGS.W, train_files, index)
    L = read_img(FLAGS.L, train_files, index, norm=False)
    ADD = X + Y + Z + W + L
    M = np.ones(L.shape, dtype="float32") * (ADD > 0.1)
    for j in range(L.shape[0]):
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(X[j, :, :]),
                             FLAGS.X_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(Y[j, :, :]),
                             FLAGS.Y_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(Z[j, :, :]),
                             FLAGS.Z_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(W[j, :, :]),
                             FLAGS.W_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(L[j, :, :]),
                             FLAGS.L_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(M[j, :, :]),
                             FLAGS.M_ + "/" + str(index) + "_" + str(j + 50) + ".tiff")


def read_filename(path):
    train_files = os.listdir(path)
    return np.asarray(train_files)


train_files = read_filename(FLAGS.L)
for i in range(len(train_files)):
    split_file(train_files, i)
