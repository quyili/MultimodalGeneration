# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('load_model', "20190822-2137", "default: None")
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_string('F_test', './F', 'files path')
tf.flags.DEFINE_string('M_test', './M', 'files path')
tf.flags.DEFINE_string('x_g', "GPU_0/DC_M/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('f_input', "GPU_0/mul_9:0", "tensor name")
tf.flags.DEFINE_string('save_path', "./test_images/", "default: ./test_images/")


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


def train():
    if FLAGS.load_model is not None:
        if FLAGS.savefile is not None:
            checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")

    else:
        logging.error("<load_model> is None.")
        return
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_path)

    graph = tf.get_default_graph()
    f_input = graph.get_tensor_by_name(FLAGS.f_input)

    x_g = tf.get_default_graph().get_tensor_by_name(FLAGS.x_g)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        try:
            os.makedirs(FLAGS.save_path + "F")
            os.makedirs(FLAGS.save_path + "M")
            os.makedirs(FLAGS.save_path + "X")
        except os.error:
            pass

        F_train_files = read_filename(FLAGS.F_test, shuffle=False)
        index = 0
        while index <= len(F_train_files):
            train_true_f = []
            for b in range(FLAGS.batch_size):
                train_F_arr_ = read_file(FLAGS.F_test, F_train_files, index).reshape(FLAGS.image_size)
                train_true_f.append(train_F_arr_)
                index = index + 1

            x_g_ = sess.run([x_g],feed_dict={f_input: np.asarray(train_true_f)})

            for b in range(FLAGS.batch_size):
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(
                    read_file(FLAGS.F_test, F_train_files, index - b - 1)
                        .reshape(FLAGS.image_size)[:, :, 0]),
                    FLAGS.save_path + "F/" + F_train_files[index - b - 1])
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(
                    read_file(FLAGS.M_test, F_train_files, index - b - 1)
                        .reshape(FLAGS.image_size)[:, :, 0]),
                    FLAGS.save_path + "M/" + F_train_files[index - b - 1])
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x_g_)[b, :, :, 0]),
                                     FLAGS.save_path + "X/" + F_train_files[index - b - 1])


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
