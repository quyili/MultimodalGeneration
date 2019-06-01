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
tf.flags.DEFINE_string('X_test', '../data/BRATS2015/testT1', 'X files for training')
tf.flags.DEFINE_string('Y_test', '../data/BRATS2015/testT2', 'Y files for training')
tf.flags.DEFINE_string('L_test', '../data/BRATS2015/testLabel', 'Y files for training')
tf.flags.DEFINE_string('M_test', '../data/BRATS2015/testMask', 'Y files for training')
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

            image_list, code_list, j_list, loss_list= gan.model()
            optimizers = gan.optimize(loss_list)

            gan.image_summary(image_list)
            gan.histogram_summary(j_list)

            evaluation_list = gan.evaluation(image_list)
            evaluation_code_list = gan.evaluation_code(code_list)

            loss_list_summary = tf.placeholder(tf.float32)
            evaluation_list_summary = tf.placeholder(tf.float32)
            evaluation_code_list_summary = tf.placeholder(tf.float32)

            gan.loss_summary(loss_list_summary)
            gan.evaluation_summary(evaluation_list_summary)
            gan.evaluation_code_summary(evaluation_code_list_summary)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir + "/train", graph)
            val_writer = tf.summary.FileWriter(checkpoints_dir + "/val", graph)
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
                Label_test_files = read_filename(FLAGS.L_test)
                index = 0
                epoch = 0
                train_loss_list = []
                train_evaluation_list = []
                train_evaluation_code_list = []
                while not coord.should_stop() and epoch <= FLAGS.epoch:
                    train_true_x = []
                    train_true_y = []
                    train_true_l = []
                    train_true_m = []
                    for b in range(FLAGS.batch_size):
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
                    _, train_losses, train_evaluations, train_evaluation_codes = sess.run(
                        [optimizers, loss_list, evaluation_list, evaluation_code_list],
                        feed_dict={
                            gan.x: np.asarray(train_true_x),
                            gan.y: np.asarray(train_true_y),
                            gan.rm: rm,
                            gan.label_expand: np.asarray(train_true_l),
                            gan.mask: np.asarray(train_true_m).astype('float32')
                        })
                    train_loss_list.append(train_losses)
                    train_evaluation_list.append(train_evaluations)
                    train_evaluation_code_list.append(train_evaluation_codes)
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")
                    
                    if step % int(FLAGS.epoch_steps/2) == 0:
                        logging.info('-----------Train summary start-------------')
                        train__summary = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(train_loss_list),
                                       evaluation_list_summary: mean_list(train_evaluation_list),
                                       evaluation_code_list_summary: mean_list(train_evaluation_code_list)})
                        train_writer.add_summary(train__summary, step)
                        train_writer.flush()
                        logging.info('-----------Train summary end-------------')

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                        logging.info(
                            "-----------val epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                        val_index=0
                        for j in range(int(len(Label_test_files)/FLAGS.batch_size)):
                            val_true_x = []
                            val_true_y = []
                            val_true_l = []
                            val_true_m = []
                            for b in range(FLAGS.batch_size):
                                val_L_arr_ = read_file(FLAGS.L, Label_test_files, val_index)
                                val_M_arr_ = read_file(FLAGS.M, Label_test_files, val_index)
                                val_X_arr_ = read_file(FLAGS.X, Label_test_files, val_index)
                                val_Y_arr_ = read_file(FLAGS.Y, Label_test_files, val_index)
                                L_arr = expand(val_M_arr_, val_L_arr_)
                                X_arr = np.asarray(val_X_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                Y_arr = np.asarray(val_Y_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                M_arr = np.asarray(val_M_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                val_true_x.append(X_arr)
                                val_true_y.append(Y_arr)
                                val_true_m.append(M_arr)
                                val_true_l.append(L_arr)
                                val_index +=1
                        
                            val_loss_list = []
                            val_evaluation_list = []
                            val_evaluation_code_list = []
                            val_image_list, val_losses, val_evaluations, val_evaluation_codes = sess.run(
                                [image_list, loss_list, evaluation_list, evaluation_code_list],
                                feed_dict={
                                    gan.x: np.asarray(val_true_x),
                                    gan.y: np.asarray(val_true_y),
                                    gan.rm: rm,
                                    gan.label_expand: np.asarray(val_true_l),
                                    gan.mask: np.asarray(val_true_m).astype('float32')
                                })
                            val_loss_list.append(val_losses)
                            val_evaluation_list.append(val_evaluations)
                            val_evaluation_code_list.append(val_evaluation_codes)

                            val__summary = sess.run(
                                summary_op,
                                feed_dict={loss_list_summary: mean_list(val_loss_list),
                                           evaluation_list_summary: mean_list(val_evaluation_list),
                                           evaluation_code_list_summary: mean_list(val_evaluation_code_list)})
                            val_writer.add_summary(val__summary, step)
                            val_writer.flush()

                            if j==0:
                                val_true_x,val_true_y, val_x_g, val_y_g, val_x_g_t, val_y_g_t, val_x_r, val_y_r, val_x_t, val_y_t, \
                                val_l_input, val_l_g, val_l_f_by_x, val_l_f_by_y, val_l_g_by_x, val_l_g_by_y = val_image_list
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_input)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/true_label_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_g)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_label_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_f_by_x)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_label_by_x_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_f_by_y)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_label_by_y_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_g_by_x)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_label_by_x_g_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_l_g_by_y)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_label_by_y_g_" + str(step) + ".tiff")

                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_true_x)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/true_x_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_true_y)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/true_y_" + str(step) + ".tiff")

                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_x_g)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_x_g_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_y_g)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_y_g_" + str(step) + ".tiff")

                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_x_g_t)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_x_g_t_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_y_g_t)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_y_g_t_" + str(step) + ".tiff")

                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_x_r)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_x_r_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_y_r)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_y_r_" + str(step) + ".tiff")

                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_x_t)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_x_t_" + str(step) + ".tiff")
                                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(val_y_t)[0, :, :, 0]),
                                                     checkpoints_dir + "/samples/fake_y_t_" + str(step) + ".tiff")
                        logging.info(
                            "-----------val epoch " + str(epoch) + ", step " + str(step) + ": end-------------")
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
