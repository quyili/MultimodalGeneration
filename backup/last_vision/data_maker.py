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

tf.flags.DEFINE_string('X_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainT1',
                       'X files for training')
tf.flags.DEFINE_string('Y_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainT1c',
                       'Y files for training')
tf.flags.DEFINE_string('Z_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainT2',
                       'Z files for training')
tf.flags.DEFINE_string('W_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainFlair',
                       'Z files for training')
tf.flags.DEFINE_string('L_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainLabel',
                       'Z files for training')
tf.flags.DEFINE_string('LV_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainLabelV',
                       'Z files for training')
tf.flags.DEFINE_string('M_', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/trainMask',
                       'Z files for training')

tf.flags.DEFINE_string('X_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testT1',
                       'X files for training')
tf.flags.DEFINE_string('Y_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testT1c',
                       'Y files for training')
tf.flags.DEFINE_string('Z_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testT2',
                       'Z files for training')
tf.flags.DEFINE_string('W_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testFlair',
                       'Z files for training')
tf.flags.DEFINE_string('L_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testLabel',
                       'Z files for training')
tf.flags.DEFINE_string('LV_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testLabelV',
                       'Z files for training')
tf.flags.DEFINE_string('M_test', '/GPUFS/nsccgz_ywang_1/quyili/data/BRATS2015/testMask',
                       'Z files for training')

os.makedirs(FLAGS.X_)
os.makedirs(FLAGS.Y_)
os.makedirs(FLAGS.Z_)
os.makedirs(FLAGS.W_)
os.makedirs(FLAGS.L_)
os.makedirs(FLAGS.LV_)
os.makedirs(FLAGS.M_)

os.makedirs(FLAGS.X_test)
os.makedirs(FLAGS.Y_test)
os.makedirs(FLAGS.Z_test)
os.makedirs(FLAGS.W_test)
os.makedirs(FLAGS.L_test)
os.makedirs(FLAGS.LV_test)
os.makedirs(FLAGS.M_test)


def read_img(path, train_files, index, norm=True):
    train_range = len(train_files)
    arr_ = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path + "/" + train_files[index % train_range])).astype(
        'float32')
    if norm:
        arr_ = (arr_ - np.min(arr_, axis=(0, 1, 2))) / (np.max(arr_, axis=(0, 1, 2)) - np.min(arr_, axis=(0, 1, 2)))
    return arr_[50:105, 38:222, 48:192]


def split_file(train_files, index, xp, yp, zp, wp, lp, lvp, mp):
    X = read_img(FLAGS.X, train_files, index)
    Y = read_img(FLAGS.Y, train_files, index)
    Z = read_img(FLAGS.Z, train_files, index)
    W = read_img(FLAGS.W, train_files, index)
    L = read_img(FLAGS.L, train_files, index, norm=False)
    ADD = X + Y + Z + W + L
    M = np.ones(L.shape, dtype="float32") * (ADD > 0.1)
    LV = (L + 1.0) * M * 0.2
    for j in range(L.shape[0]):
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(X[j, :, :]),
                             xp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(Y[j, :, :]),
                             yp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(Z[j, :, :]),
                             zp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(W[j, :, :]),
                             wp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(L[j, :, :]),
                             lp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(LV[j, :, :]),
                             lvp + "/" + str(index) + "_" + str(j + 50) + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(M[j, :, :]),
                             mp + "/" + str(index) + "_" + str(j + 50) + ".tiff")


def read_filename(path, rate=0.0, shuffle=True):
    files = os.listdir(path)
    train_range = int(len(files) * rate)
    files=np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files))
        np.random.shuffle(index_arr)
        files = files[index_arr]
    train_files = files[:train_range]
    val_files = files[train_range:]
    return np.asarray(train_files), np.asarray(val_files)


train_files, val_files = read_filename(FLAGS.L, rate=0.891)
for i in range(len(train_files)):
    split_file(train_files, i, FLAGS.X_, FLAGS.Y_, FLAGS.Z_, FLAGS.W_, FLAGS.L_, FLAGS.LV_, FLAGS.M_)
for j in range(len(val_files)):
    split_file(train_files, j, FLAGS.X_test, FLAGS.Y_test, FLAGS.Z_test, FLAGS.W_test, FLAGS.L_test, FLAGS.LV_test,
               FLAGS.M_test)
