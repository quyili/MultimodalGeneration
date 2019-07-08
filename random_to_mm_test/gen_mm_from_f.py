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
tf.flags.DEFINE_string('checkpoint_dir', "./checkpoints/20190621-1650/", "default: None")
tf.flags.DEFINE_string('meta_dir', "model.ckpt-137556.meta", "default: None")
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_integer('epoch_steps', 15070, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('F_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/done/test46/test2',
                       'X files for training')
tf.flags.DEFINE_string('L_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata//BRATS2015/testLabel',
                       'Y files for training')
tf.flags.DEFINE_string('M_test', '/GPUFS/nsccgz_ywang_1/quyili/MultimodalGeneration/mydata/BRATS2015/testMask',
                       'Y files for training')
tf.flags.DEFINE_string('x_g', "GPU_0/DC_X/lastconv/Sigmoid:0", "default: None")
tf.flags.DEFINE_string('y_g', "GPU_0/DC_Y/lastconv/Sigmoid:0", "default: None")
tf.flags.DEFINE_string('z_g', "GPU_0/DC_Z/lastconv/Sigmoid:0", "default: None")
tf.flags.DEFINE_string('w_g', "GPU_0/DC_W/lastconv/Sigmoid:0", "default: None")
tf.flags.DEFINE_string('f_input', "GPU_0/mul_1:0", "default: None")
tf.flags.DEFINE_string('L_input', "GPU_0/Placeholder_2:0", "default: None")


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
    saver = tf.train.import_meta_graph(FLAGS.checkpoints_dir + FLAGS.meta_dir)

    f_input = graph.get_tensor_by_name(FLAGS.f_input)
    l_input = graph.get_tensor_by_name(FLAGS.l_input)

    x_g = tf.get_default_graph().get_tensor_by_name(FLAGS.x_g)
    y_g = tf.get_default_graph().get_tensor_by_name(FLAGS.y_g)
    z_g = tf.get_default_graph().get_tensor_by_name(FLAGS.z_g)
    w_g = tf.get_default_graph().get_tensor_by_name(FLAGS.w_g)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoints_dir))
        try:
            os.makedirs("./test_images/X")
            os.makedirs("./test_images/Y")
            os.makedirs("./test_images/Z")
            os.makedirs("./test_images/W")
            os.makedirs("./test_images/L")
        except os.error:
            pass

        F_train_files = read_filename(FLAGS.F_test)
        index = 0
        while index <= FLAGS.epoch_steps:
            train_true_f = []
            train_true_l = []
            for b in range(1):
                train_F_arr_ = read_file(FLAGS.F_test, F_train_files, index).reshape(FLAGS.image_size)
                train_Mask_arr_ = read_file(FLAGS.Mask_test, F_train_files, index).reshape(FLAGS.image_size)
                while True:
                    train_L_arr_ = read_file(FLAGS.L_test, F_train_files,
                                             np.random.randint(len(F_train_files))).reshape(FLAGS.image_size)
                    if np.sum(train_Mask_arr_ * train_L_arr_) == 0.0: break
                    logging.info("mask and label not match !")

                train_true_f.append(train_F_arr_)
                train_true_l.append(train_L_arr_)
                index = index + 1

            print("image gen start:" + str(index))
            x_g_, y_g_, z_g_, w_g_ = sess.run([x_g, y_g, z_g, w_g],
                                              feed_dict={f_input: np.asarray(train_true_f),
                                                         l_input: np.asarray(train_true_l)})

            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x_g_)[0, :, :, 0]),
                                 "./test_images/X/fake_x_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y_g_)[0, :, :, 0]),
                                 "./test_images/Y/fake_y_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(z_g_)[0, :, :, 0]),
                                 "./test_images/Z/fake_z_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(w_g_)[0, :, :, 0]),
                                 "./test_images/W/fake_w_" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_true_l)[0, :, :, 0]),
                                 "./test_images/L/fake_l_" + str(index) + ".tiff")

            print("image gen end:" + str(index))


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
