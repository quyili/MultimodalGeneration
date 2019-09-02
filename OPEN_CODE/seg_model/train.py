# _*_ coding:utf-8 _*_
import tensorflow as tf
from seg_model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK
import math

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 1')
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', '../../mydata/BRATS2015/trainT1', 'X files for training')
tf.flags.DEFINE_string('Y', '../../mydata/BRATS2015/trainT2', 'Y files for training')
tf.flags.DEFINE_string('Z', '../../mydata/BRATS2015/trainT1c', 'X files for training')
tf.flags.DEFINE_string('W', '../../mydata/BRATS2015/trainFlair', 'Y files for training')
tf.flags.DEFINE_string('L', '../../mydata/BRATS2015/trainLabel', 'Y files for training')
tf.flags.DEFINE_string('X_test', '../../mydata/BRATS2015/testT1', 'X files for training')
tf.flags.DEFINE_string('Y_test', '../../mydata/BRATS2015/testT2', 'Y files for training')
tf.flags.DEFINE_string('Z_test', '../../mydata/BRATS2015/testT1c', 'X files for training')
tf.flags.DEFINE_string('W_test', '../../mydata/BRATS2015/testFlair', 'Y files for training')
tf.flags.DEFINE_string('L_test', '../../mydata/BRATS2015/testLabel', 'Y files for training')
tf.flags.DEFINE_string('load_model', "20190801-1128",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 100, 'default: 100')
tf.flags.DEFINE_float('display_epoch', 1, 'default: 1')
tf.flags.DEFINE_integer('epoch_steps', 15070, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('stage', "test", 'default: train')
tf.flags.DEFINE_integer('select_num', 160001, 'default: 15070')
tf.flags.DEFINE_float('select_score1', 0.0, 'default: 0.9')
tf.flags.DEFINE_string('select_save_path', './', 'default: ./')


def mean(list):
    return sum(list) / float(len(list))


def mean_list(lists):
    out = []
    lists = np.asarray(lists).transpose([1, 0])
    for list in lists:
        out.append(mean(list))
    return out


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return np.asarray(L_arr_)


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def average_gradients(grads_list):
    average_grads = []
    for grad_and_vars in zip(*grads_list):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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
            except os.error:
                pass

        for attr, value in FLAGS.flag_values_dict().items():
            logging.info("%s\t:\t%s" % (attr, str(value)))

        graph = tf.Graph()
        with graph.as_default():
            gan = GAN(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.ngf)
            input_shape = [int(FLAGS.batch_size / 4), FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]]
            G_optimizer = gan.optimize()

            G_grad_list = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/gpu:0"):
                    with tf.name_scope("GPU_0"):
                        l_x_0 = tf.placeholder(tf.float32, shape=input_shape)
                        l_y_0 = tf.placeholder(tf.float32, shape=input_shape)
                        l_z_0 = tf.placeholder(tf.float32, shape=input_shape)
                        l_w_0 = tf.placeholder(tf.float32, shape=input_shape)
                        x_0 = tf.placeholder(tf.float32, shape=input_shape)
                        y_0 = tf.placeholder(tf.float32, shape=input_shape)
                        z_0 = tf.placeholder(tf.float32, shape=input_shape)
                        w_0 = tf.placeholder(tf.float32, shape=input_shape)
                        G_loss_0 = gan.model(l_x_0, l_y_0, l_z_0, l_w_0, x_0, y_0, z_0, w_0)
                        image_list_0 = gan.image_list
                        evaluation_list_0, mse_list_0 = gan.evaluation(image_list_0)
                        variables_list_0 = gan.get_variables()
                        G_grad_0 = G_optimizer.compute_gradients(G_loss_0, var_list=variables_list_0[0])
                        G_grad_list.append(G_grad_0)
                with tf.device("/gpu:1"):
                    with tf.name_scope("GPU_1"):
                        l_x_1 = tf.placeholder(tf.float32, shape=input_shape)
                        l_y_1 = tf.placeholder(tf.float32, shape=input_shape)
                        l_z_1 = tf.placeholder(tf.float32, shape=input_shape)
                        l_w_1 = tf.placeholder(tf.float32, shape=input_shape)
                        x_1 = tf.placeholder(tf.float32, shape=input_shape)
                        y_1 = tf.placeholder(tf.float32, shape=input_shape)
                        z_1 = tf.placeholder(tf.float32, shape=input_shape)
                        w_1 = tf.placeholder(tf.float32, shape=input_shape)
                        G_loss_1 = gan.model(l_x_1, l_y_1, l_z_1, l_w_1, x_1, y_1, z_1, w_1)
                        image_list_1 = gan.image_list
                        evaluation_list_1, mse_list_1 = gan.evaluation(image_list_1)
                        variables_list_1 = gan.get_variables()
                        G_grad_1 = G_optimizer.compute_gradients(G_loss_1, var_list=variables_list_1[0])
                        G_grad_list.append(G_grad_1)
                with tf.device("/gpu:2"):
                    with tf.name_scope("GPU_2"):
                        l_x_2 = tf.placeholder(tf.float32, shape=input_shape)
                        l_y_2 = tf.placeholder(tf.float32, shape=input_shape)
                        l_z_2 = tf.placeholder(tf.float32, shape=input_shape)
                        l_w_2 = tf.placeholder(tf.float32, shape=input_shape)
                        x_2 = tf.placeholder(tf.float32, shape=input_shape)
                        y_2 = tf.placeholder(tf.float32, shape=input_shape)
                        z_2 = tf.placeholder(tf.float32, shape=input_shape)
                        w_2 = tf.placeholder(tf.float32, shape=input_shape)
                        G_loss_2 = gan.model(l_x_2, l_y_2, l_z_2, l_w_2, x_2, y_2, z_2, w_2)
                        image_list_2 = gan.image_list
                        evaluation_list_2, mse_list_2 = gan.evaluation(image_list_2)
                        variables_list_2 = gan.get_variables()
                        G_grad_2 = G_optimizer.compute_gradients(G_loss_2, var_list=variables_list_2[0])
                        G_grad_list.append(G_grad_2)
                with tf.device("/gpu:3"):
                    with tf.name_scope("GPU_3"):
                        l_x_3 = tf.placeholder(tf.float32, shape=input_shape)
                        l_y_3 = tf.placeholder(tf.float32, shape=input_shape)
                        l_z_3 = tf.placeholder(tf.float32, shape=input_shape)
                        l_w_3 = tf.placeholder(tf.float32, shape=input_shape)
                        x_3 = tf.placeholder(tf.float32, shape=input_shape)
                        y_3 = tf.placeholder(tf.float32, shape=input_shape)
                        z_3 = tf.placeholder(tf.float32, shape=input_shape)
                        w_3 = tf.placeholder(tf.float32, shape=input_shape)
                        G_loss_3 = gan.model(l_x_3, l_y_3, l_z_3, l_w_3, x_3, y_3, z_3, w_3)
                        image_list_3 = gan.image_list
                        evaluation_list_3, mse_list_3 = gan.evaluation(image_list_3)
                        variables_list_3 = gan.get_variables()
                        G_grad_3 = G_optimizer.compute_gradients(G_loss_3, var_list=variables_list_3[0])
                        G_grad_list.append(G_grad_3)

            G_ave_grad = average_gradients(G_grad_list)
            optimizers = G_optimizer.apply_gradients(G_ave_grad)
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
                if FLAGS.stage == "train":
                    l_x_train_files = read_filename(FLAGS.L)
                    l_y_train_files = read_filename(FLAGS.L)
                    l_z_train_files = read_filename(FLAGS.L)
                    l_w_train_files = read_filename(FLAGS.L)
                    index = 0
                    epoch = 0
                    while not coord.should_stop() and epoch <= FLAGS.epoch:

                        train_true_l_x = []
                        train_true_l_y = []
                        train_true_l_z = []
                        train_true_l_w = []
                        train_true_x = []
                        train_true_y = []
                        train_true_z = []
                        train_true_w = []
                        for b in range(FLAGS.batch_size):
                            train_l_x_arr = read_file(FLAGS.L, l_x_train_files, index).reshape(FLAGS.image_size)
                            train_x_arr = read_file(FLAGS.X, l_x_train_files, index).reshape(FLAGS.image_size)
                            train_l_y_arr = read_file(FLAGS.L, l_y_train_files, index).reshape(FLAGS.image_size)
                            train_y_arr = read_file(FLAGS.Y, l_y_train_files, index).reshape(FLAGS.image_size)
                            train_l_z_arr = read_file(FLAGS.L, l_z_train_files, index).reshape(FLAGS.image_size)
                            train_z_arr = read_file(FLAGS.Z, l_z_train_files, index).reshape(FLAGS.image_size)
                            train_l_w_arr = read_file(FLAGS.L, l_w_train_files, index).reshape(FLAGS.image_size)
                            train_w_arr = read_file(FLAGS.W, l_w_train_files, index).reshape(FLAGS.image_size)

                            train_true_l_x.append(train_l_x_arr)
                            train_true_l_y.append(train_l_y_arr)
                            train_true_l_z.append(train_l_z_arr)
                            train_true_l_w.append(train_l_w_arr)
                            train_true_x.append(train_x_arr)
                            train_true_y.append(train_y_arr)
                            train_true_z.append(train_z_arr)
                            train_true_w.append(train_w_arr)

                            epoch = int(index / len(l_x_train_files))
                            index = index + 1

                        logging.info(
                            "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                        sess.run(
                            optimizers,
                            feed_dict={
                                l_x_0: np.asarray(train_true_l_x)[0:1, :, :, :],
                                l_y_0: np.asarray(train_true_l_y)[0:1, :, :, :],
                                l_z_0: np.asarray(train_true_l_z)[0:1, :, :, :],
                                l_w_0: np.asarray(train_true_l_w)[0:1, :, :, :],
                                x_0: np.asarray(train_true_x)[0:1, :, :, :],
                                y_0: np.asarray(train_true_y)[0:1, :, :, :],
                                z_0: np.asarray(train_true_z)[0:1, :, :, :],
                                w_0: np.asarray(train_true_w)[0:1, :, :, :],

                                l_x_1: np.asarray(train_true_l_x)[1:2, :, :, :],
                                l_y_1: np.asarray(train_true_l_y)[1:2, :, :, :],
                                l_z_1: np.asarray(train_true_l_z)[1:2, :, :, :],
                                l_w_1: np.asarray(train_true_l_w)[1:2, :, :, :],
                                x_1: np.asarray(train_true_x)[1:2, :, :, :],
                                y_1: np.asarray(train_true_y)[1:2, :, :, :],
                                z_1: np.asarray(train_true_z)[1:2, :, :, :],
                                w_1: np.asarray(train_true_w)[1:2, :, :, :],

                                l_x_2: np.asarray(train_true_l_x)[2:3, :, :, :],
                                l_y_2: np.asarray(train_true_l_y)[2:3, :, :, :],
                                l_z_2: np.asarray(train_true_l_z)[2:3, :, :, :],
                                l_w_2: np.asarray(train_true_l_w)[2:3, :, :, :],
                                x_2: np.asarray(train_true_x)[2:3, :, :, :],
                                y_2: np.asarray(train_true_y)[2:3, :, :, :],
                                z_2: np.asarray(train_true_z)[2:3, :, :, :],
                                w_2: np.asarray(train_true_w)[2:3, :, :, :],

                                l_x_3: np.asarray(train_true_l_x)[3:4, :, :, :],
                                l_y_3: np.asarray(train_true_l_y)[3:4, :, :, :],
                                l_z_3: np.asarray(train_true_l_z)[3:4, :, :, :],
                                l_w_3: np.asarray(train_true_l_w)[3:4, :, :, :],
                                x_3: np.asarray(train_true_x)[3:4, :, :, :],
                                y_3: np.asarray(train_true_y)[3:4, :, :, :],
                                z_3: np.asarray(train_true_z)[3:4, :, :, :],
                                w_3: np.asarray(train_true_w)[3:4, :, :, :],
                            })
                        logging.info(
                            "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")
                        step += 1

                elif FLAGS.stage == "test":
                    val_evaluation_list = []
                    val_mse_list = []
                    val_index = 0
                    count = 0
                    select_files = []

                    l_val_files = read_filename(FLAGS.L_test, shuffle=False)
                    for j in range(int(math.ceil(len(l_val_files) / FLAGS.batch_size))):
                        val_true_l = []
                        val_true_x = []
                        val_true_y = []
                        val_true_z = []
                        val_true_w = []
                        for b in range(FLAGS.batch_size):
                            val_l_arr = read_file(FLAGS.L_test, l_val_files, val_index).reshape(FLAGS.image_size)
                            val_x_arr = read_file(FLAGS.X_test, l_val_files, val_index).reshape(FLAGS.image_size)
                            val_y_arr = read_file(FLAGS.Y_test, l_val_files, val_index).reshape(FLAGS.image_size)
                            val_z_arr = read_file(FLAGS.Z_test, l_val_files, val_index).reshape(FLAGS.image_size)
                            val_w_arr = read_file(FLAGS.W_test, l_val_files, val_index).reshape(FLAGS.image_size)

                            val_true_l.append(val_l_arr)
                            val_true_x.append(val_x_arr)
                            val_true_y.append(val_y_arr)
                            val_true_z.append(val_z_arr)
                            val_true_w.append(val_w_arr)

                            val_index += 1

                        val_evaluations_0, val_mses_0, \
                        val_evaluations_1, val_mses_1, \
                        val_evaluations_2, val_mses_2, \
                        val_evaluations_3, val_mses_3 = sess.run(
                            [evaluation_list_0, mse_list_0,
                             evaluation_list_1, mse_list_1,
                             evaluation_list_2, mse_list_2,
                             evaluation_list_3, mse_list_3],
                            feed_dict={
                                l_x_0: np.asarray(val_true_l)[0:1, :, :, :],
                                l_y_0: np.asarray(val_true_l)[0:1, :, :, :],
                                l_z_0: np.asarray(val_true_l)[0:1, :, :, :],
                                l_w_0: np.asarray(val_true_l)[0:1, :, :, :],
                                x_0: np.asarray(val_true_x)[0:1, :, :, :],
                                y_0: np.asarray(val_true_y)[0:1, :, :, :],
                                z_0: np.asarray(val_true_z)[0:1, :, :, :],
                                w_0: np.asarray(val_true_w)[0:1, :, :, :],

                                l_x_1: np.asarray(val_true_l)[1:2, :, :, :],
                                l_y_1: np.asarray(val_true_l)[1:2, :, :, :],
                                l_z_1: np.asarray(val_true_l)[1:2, :, :, :],
                                l_w_1: np.asarray(val_true_l)[1:2, :, :, :],
                                x_1: np.asarray(val_true_x)[1:2, :, :, :],
                                y_1: np.asarray(val_true_y)[1:2, :, :, :],
                                z_1: np.asarray(val_true_z)[1:2, :, :, :],
                                w_1: np.asarray(val_true_w)[1:2, :, :, :],

                                l_x_2: np.asarray(val_true_l)[2:3, :, :, :],
                                l_y_2: np.asarray(val_true_l)[2:3, :, :, :],
                                l_z_2: np.asarray(val_true_l)[2:3, :, :, :],
                                l_w_2: np.asarray(val_true_l)[2:3, :, :, :],
                                x_2: np.asarray(val_true_x)[2:3, :, :, :],
                                y_2: np.asarray(val_true_y)[2:3, :, :, :],
                                z_2: np.asarray(val_true_z)[2:3, :, :, :],
                                w_2: np.asarray(val_true_w)[2:3, :, :, :],

                                l_x_3: np.asarray(val_true_l)[3:4, :, :, :],
                                l_y_3: np.asarray(val_true_l)[3:4, :, :, :],
                                l_z_3: np.asarray(val_true_l)[3:4, :, :, :],
                                l_w_3: np.asarray(val_true_l)[3:4, :, :, :],
                                x_3: np.asarray(val_true_x)[3:4, :, :, :],
                                y_3: np.asarray(val_true_y)[3:4, :, :, :],
                                z_3: np.asarray(val_true_z)[3:4, :, :, :],
                                w_3: np.asarray(val_true_w)[3:4, :, :, :],
                            })

                        if mean(val_evaluations_0) <= FLAGS.select_score2 \
                                and mean(val_evaluations_0) > FLAGS.select_score1 \
                                and count < FLAGS.select_num:
                            val_evaluation_list.append(val_evaluations_0)
                            val_mse_list.append(val_mses_0)
                            select_files.append(l_val_files[j * 4 + 0])
                            count += 1

                        if mean(val_evaluations_1) <= FLAGS.select_score2 \
                                and mean(val_evaluations_1) > FLAGS.select_score1 \
                                and count < FLAGS.select_num:
                            val_evaluation_list.append(val_evaluations_1)
                            val_mse_list.append(val_mses_1)
                            select_files.append(l_val_files[j * 4 + 1])
                            count += 1

                        if mean(val_evaluations_2) < FLAGS.select_score2 \
                                and mean(val_evaluations_2) >= FLAGS.select_score1 \
                                and count < FLAGS.select_num:
                            val_evaluation_list.append(val_evaluations_2)
                            val_mse_list.append(val_mses_2)
                            select_files.append(l_val_files[j * 4 + 2])
                            count += 1

                        if mean(val_evaluations_3) < FLAGS.select_score2 \
                                and mean(val_evaluations_3) >= FLAGS.select_score1 \
                                and count < FLAGS.select_num:
                            val_evaluation_list.append(val_evaluations_3)
                            val_mse_list.append(val_mses_3)
                            select_files.append(l_val_files[j * 4 + 3])
                            count += 1

                        if count >= FLAGS.select_num:
                            break

                    print("MSE:", mean_list(val_mse_list), mean(mean_list(val_mse_list)))
                    print("MEAN Dice Score:", mean_list(val_evaluation_list), mean(mean_list(val_evaluation_list)))
                    print("select_files:", select_files)

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
