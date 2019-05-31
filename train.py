# _*_ coding:utf-8 _*_
import tensorflow as tf
from gen_model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('devices', "0,1,2,3", 'tf gpu')
tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_float('learning_rate', 2e-5, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', '../data/BRATS2015/trainT1', 'X files for training')
tf.flags.DEFINE_string('Y', '../data/BRATS2015/trainT2', 'Y files for training')
tf.flags.DEFINE_string('L', '../data/BRATS2015/trainLabel', 'Y files for training')
tf.flags.DEFINE_string('M', '../data/BRATS2015/trainMask', 'Y files for training')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 10, 'default: 100')
tf.flags.DEFINE_float('display_epoch', 1, 'default: 1')
tf.flags.DEFINE_integer('epoch_steps', 15070, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('stage', "train", 'default: train')


def mean(list):
    return sum(list) / float(len(list))


def mean_list(lists):
    out = []
    lists = np.asarray(lists).transpose([1, 0])
    for list in lists:
        out.append(mean(list))
    return out


def random(n, h, w, c):
    return np.random.uniform(0., 1., size=[n, h, w, c])


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


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def train():
    with tf.device("/cpu:0"):
        if FLAGS.load_model is not None:
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
            else:
                checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/{}".format(current_time)
            else:
                checkpoints_dir = "checkpoints/{}".format(current_time)
            try:
                os.makedirs(checkpoints_dir + "/samples")
                os.makedirs(checkpoints_dir + "/best")
            except os.error:
                pass

        for attr, value in FLAGS.flag_values_dict().items():
            logging.info("%s\t:\t%s" % (attr, str(value)))

        graph = tf.Graph()
        with graph.as_default():
            gan = GAN(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.ngf)

            l_input, rm_input, x_g, y_g, x_r, y_r, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y, \
            G_loss, D_loss, evluation_list = gan.model()

            optimizers = gan.optimize(G_loss, D_loss)
            saver = tf.train.Saver()

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if FLAGS.load_model is not None:
                logging.info("restore model:" + FLAGS.load_model)
                if FLAGS.checkpoint is not None:
                    model_checkpoint_path = checkpoints_dir + "/model.ckpt-" + FLAGS.checkpoint
                    latest_checkpoint = model_checkpoint_path
                else:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    model_checkpoint_path = checkpoint.model_checkpoint_path
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
                logging.info("model checkpoint path:" + model_checkpoint_path)
                meta_graph_path = model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, latest_checkpoint)
                if FLAGS.step_clear == True:
                    step = 0
                else:
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0
            sess.graph.finalize()
            logging.info("start step:" + str(step))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                Label_train_files = read_filename(FLAGS.L)
                index = 0
                epoch = 0
                while not coord.should_stop() and epoch <= FLAGS.epoch:
                    train_true_x = []
                    train_true_y = []
                    train_true_l = []
                    train_true_m = []
                    for i in range(FLAGS.batch_size):
                        train_L_arr_ = read_file(FLAGS.L, Label_train_files, index)
                        train_M_arr_ = read_file(FLAGS.M, Label_train_files, index)
                        train_X_arr_ = read_file(FLAGS.X, Label_train_files, index)
                        train_Y_arr_ = read_file(FLAGS.Y, Label_train_files, index)
                        L_arr = expand(train_M_arr_, train_L_arr_)
                        X_arr = np.asarray(train_X_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        Y_arr = np.asarray(train_Y_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        M_arr = np.asarray(train_M_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        train_true_x.append(X_arr)
                        train_true_y.append(Y_arr)
                        train_true_m.append(M_arr)
                        train_true_l.append(L_arr)
                        epoch = int(index / len(Label_train_files))
                        index = index + 1

                    rm = random(FLAGS.batch_size, FLAGS.image_size[0], FLAGS.image_size[1],
                                FLAGS.image_size[2]).astype('float32')
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    _, train_l_input, train_rm_input, train_x_g, train_y_g, train_x_r, train_y_r, \
                    train_l_g, train_l_f_by_x, train_l_f_by_y, train_l_g_by_x, train_l_g_by_y, \
                    train_evluation_list = sess.run(
                        [optimizers, l_input, rm_input, x_g, y_g, x_r, y_r, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y,
                         evluation_list],
                        feed_dict={
                            gan.x: np.asarray(train_true_x),
                            gan.y: np.asarray(train_true_y),
                            gan.rm: rm,
                            gan.label_expand: np.asarray(train_true_l),
                            gan.mask: np.asarray(train_true_m).astype('float32')
                        })
                    logging.info("train_evluation_list:" + str(train_evluation_list))
                    logging.info("-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    if step % 7500 == 0:
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_input)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/true_label_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_g)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_label_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_f_by_x)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_label_by_x_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_f_by_y)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_label_by_y_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_g_by_x)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_label_by_x_g_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_l_g_by_y)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_label_by_y_g_" + str(step) + ".tiff")

                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_true_x)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/true_x_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_x_g)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_x_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_x_r)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_x_r_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_true_y)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/true_y_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_y_g)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_y_" + str(step) + ".tiff")
                        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(train_y_r)[0, :, :, 0]),
                                             checkpoints_dir + "/samples/fake_y_r_" + str(step) + ".tiff")
                    step += 1
            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
