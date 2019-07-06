# _*_ coding:utf-8 _*_
import tensorflow as tf
# from GAN_test_model import GAN
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_string('load_model', "20190620-2035",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_integer('epoch_steps', 15070, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('F_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/done/test46/test2',
                       'X files for training')
tf.flags.DEFINE_string('L_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata//BRATS2015/testLabel',
                       'Y files for training')
tf.flags.DEFINE_string('M_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/testMask',
                       'Y files for training')


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return L_arr_


def expand(train_M_arr_, train_L_arr_):
    L0 = np.asarray(train_M_arr_ == 0., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L1 = (np.asarray(train_L_arr_ == 0., "float32") * np.asarray(train_M_arr_).astype('float32')).reshape(
        [train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L2 = np.asarray(train_L_arr_ == 1., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L3 = np.asarray(train_L_arr_ == 2., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L4 = np.asarray(train_L_arr_ == 3., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L5 = np.asarray(train_L_arr_ == 4., "float32").reshape([train_L_arr_.shape[0], train_L_arr_.shape[1], 1])
    L_arr = np.concatenate([L0, L1, L2, L3, L4, L5], axis=-1)
    return L_arr


def read_files(x_path, l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    T1_img = SimpleITK.ReadImage(x_path + "/" + Label_train_files[index % train_range])
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    T1_arr_ = SimpleITK.GetArrayFromImage(T1_img)
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    T1_arr_ = T1_arr_.astype('float32')
    L_arr_ = L_arr_.astype('float32')
    return T1_arr_, L_arr_


def train():
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph("./checkpoints/20190621-1650/model.ckpt-137556.meta")

    f_input = graph.get_tensor_by_name("GPU_0/mul_1:0")
    label_expand_input = graph.get_tensor_by_name("GPU_0/Placeholder_2:0")

    x_g = tf.get_default_graph().get_tensor_by_name("GPU_0/DC_X/lastconv/Sigmoid:0")
    y_g = tf.get_default_graph().get_tensor_by_name("GPU_0/DC_Y/lastconv/Sigmoid:0")

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        checkpoints_dir = "./checkpoints/20190621-1650/"
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))

        L_train_files = read_filename(FLAGS.L_test)
        F_train_files = read_filename(FLAGS.F_test)
        index = 0
        while index <= FLAGS.epoch_steps:
            train_true_f = []
            train_true_l = []
            for b in range(1):
                train_F_arr_ = read_file(FLAGS.F_test, F_train_files, index)
                train_L_arr_ = read_file(FLAGS.L_test, L_train_files, index)
                train_M_arr_ = read_file(FLAGS.M_test, L_train_files, index)
                L_arr = expand(train_M_arr_, train_L_arr_)
                F_arr = np.asarray(train_F_arr_).reshape(
                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))

                train_true_f.append(F_arr)
                train_true_l.append(L_arr)
                index = index + 1

            print("image gen start:" + str(index))
            x_g_, y_g_ = sess.run([x_g, y_g],
                                  feed_dict={f_input: np.asarray(train_true_f),
                                             label_expand_input: np.asarray(train_true_l)})

            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x_g_)[0, :, :, 0]),
                                 "./test/fake_x_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y_g_)[0, :, :, 0]),
                                 "./test/fake_y_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_true_l)[0, :, :, 0]),
                                 "./test/fake_l_" + str(index) + ".tiff")

            print("image gen end:" + str(index))


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
