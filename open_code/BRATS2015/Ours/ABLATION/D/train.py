# _*_ coding:utf-8 _*_
import tensorflow as tf
from model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 4')
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size,')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for Adam, default: 1e-4')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', '../../mydata/BRATS2015/trainT1', 'files path')
tf.flags.DEFINE_string('Y', '../../mydata/BRATS2015/trainT2', 'files path')
tf.flags.DEFINE_string('Z', '../../mydata/BRATS2015/trainT1c', 'files path')
tf.flags.DEFINE_string('W', '../../mydata/BRATS2015/trainFlair', 'files path')
tf.flags.DEFINE_string('L', '../../mydata/BRATS2015/trainLabel', 'files path')
tf.flags.DEFINE_string('X_test', '../../mydata/BRATS2015/testT1', 'files path')
tf.flags.DEFINE_string('Y_test', '../../mydata/BRATS2015/testT2', 'files path')
tf.flags.DEFINE_string('Z_test', '../../mydata/BRATS2015/testT1c', 'files path')
tf.flags.DEFINE_string('W_test', '../../mydata/BRATS2015/testFlair', 'files path')
tf.flags.DEFINE_string('L_test', '../../mydata/BRATS2015/testLabel', 'files path')
tf.flags.DEFINE_string('load_model', "20190822-2137",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 10, 'default: 100')


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
            G_optimizer, D_optimizer, S_optimizer = gan.optimize()

            G_grad_list = []
            D_grad_list = []
            S_grad_list = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/gpu:0"):
                    with tf.name_scope("GPU_0"):
                        l_0 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_0 = tf.placeholder(tf.float32, shape=input_shape)
                        m_0 = tf.placeholder(tf.float32, shape=input_shape)
                        x_0 = tf.placeholder(tf.float32, shape=input_shape)
                        y_0 = tf.placeholder(tf.float32, shape=input_shape)
                        z_0 = tf.placeholder(tf.float32, shape=input_shape)
                        w_0 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_0 = gan.model(l_0, l_m_0, m_0, x_0, y_0, z_0, w_0)
                        tensor_name_dirct_0 = gan.tenaor_name
                        variables_list_0 = gan.get_variables()
                        G_grad_0 = G_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                        D_grad_0 = D_optimizer.compute_gradients(loss_list_0[1], var_list=variables_list_0[1])
                        S_grad_0 = D_optimizer.compute_gradients(loss_list_0[2], var_list=variables_list_0[2])
                        G_grad_list.append(G_grad_0)
                        D_grad_list.append(D_grad_0)
                        S_grad_list.append(S_grad_0)
                with tf.device("/gpu:1"):
                    with tf.name_scope("GPU_1"):
                        l_1 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_1 = tf.placeholder(tf.float32, shape=input_shape)
                        m_1 = tf.placeholder(tf.float32, shape=input_shape)
                        x_1 = tf.placeholder(tf.float32, shape=input_shape)
                        y_1 = tf.placeholder(tf.float32, shape=input_shape)
                        z_1 = tf.placeholder(tf.float32, shape=input_shape)
                        w_1 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_1 = gan.model(l_1, l_m_1, m_1, x_1, y_1, z_1, w_1)
                        variables_list_1 = gan.get_variables()
                        G_grad_1 = G_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                        D_grad_1 = D_optimizer.compute_gradients(loss_list_1[1], var_list=variables_list_1[1])
                        S_grad_1 = D_optimizer.compute_gradients(loss_list_1[2], var_list=variables_list_1[2])
                        G_grad_list.append(G_grad_1)
                        D_grad_list.append(D_grad_1)
                        S_grad_list.append(S_grad_1)
                with tf.device("/gpu:2"):
                    with tf.name_scope("GPU_2"):
                        l_2 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_2 = tf.placeholder(tf.float32, shape=input_shape)
                        m_2 = tf.placeholder(tf.float32, shape=input_shape)
                        x_2 = tf.placeholder(tf.float32, shape=input_shape)
                        y_2 = tf.placeholder(tf.float32, shape=input_shape)
                        z_2 = tf.placeholder(tf.float32, shape=input_shape)
                        w_2 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_2 = gan.model(l_2, l_m_2, m_2, x_2, y_2, z_2, w_2)
                        variables_list_2 = gan.get_variables()
                        G_grad_2 = G_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                        D_grad_2 = D_optimizer.compute_gradients(loss_list_2[1], var_list=variables_list_2[1])
                        S_grad_2 = D_optimizer.compute_gradients(loss_list_2[2], var_list=variables_list_2[2])
                        G_grad_list.append(G_grad_2)
                        D_grad_list.append(D_grad_2)
                        S_grad_list.append(S_grad_2)
                with tf.device("/gpu:3"):
                    with tf.name_scope("GPU_3"):
                        l_3 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_3 = tf.placeholder(tf.float32, shape=input_shape)
                        m_3 = tf.placeholder(tf.float32, shape=input_shape)
                        x_3 = tf.placeholder(tf.float32, shape=input_shape)
                        y_3 = tf.placeholder(tf.float32, shape=input_shape)
                        z_3 = tf.placeholder(tf.float32, shape=input_shape)
                        w_3 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_3 = gan.model(l_3, l_m_3, m_3, x_3, y_3, z_3, w_3)
                        variables_list_3 = gan.get_variables()
                        G_grad_3 = G_optimizer.compute_gradients(loss_list_3[0], var_list=variables_list_3[0])
                        D_grad_3 = D_optimizer.compute_gradients(loss_list_3[1], var_list=variables_list_3[1])
                        S_grad_3 = D_optimizer.compute_gradients(loss_list_3[2], var_list=variables_list_3[2])
                        G_grad_list.append(G_grad_3)
                        D_grad_list.append(D_grad_3)
                        S_grad_list.append(S_grad_3)

            G_ave_grad = average_gradients(G_grad_list)
            D_ave_grad = average_gradients(D_grad_list)
            S_ave_grad = average_gradients(S_grad_list)
            G_optimizer_op = G_optimizer.apply_gradients(G_ave_grad)
            D_optimizer_op = D_optimizer.apply_gradients(D_ave_grad)
            S_optimizer_op = S_optimizer.apply_gradients(S_ave_grad)
            optimizers = [G_optimizer_op, D_optimizer_op, S_optimizer_op]

            saver = tf.train.Saver()
            variables_list = gan.get_variables()

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
                logging.info("tensor_name_dirct:\n" + str(tensor_name_dirct_0))
                l_train_files = read_filename(FLAGS.L)
                l_x_train_files = read_filename(FLAGS.L)
                l_y_train_files = read_filename(FLAGS.L)
                l_z_train_files = read_filename(FLAGS.L)
                l_w_train_files = read_filename(FLAGS.L)
                index = 0
                epoch = 0
                while not coord.should_stop() and epoch <= FLAGS.epoch:

                    train_true_l = []
                    train_true_l_m = []
                    train_true_m = []
                    train_true_x = []
                    train_true_y = []
                    train_true_z = []
                    train_true_w = []
                    for b in range(FLAGS.batch_size):
                        train_m_arr = read_file(np.asarray([FLAGS.X, FLAGS.Y, FLAGS.Z, FLAGS.W])[np.random.randint(4)],
                                                l_train_files, index).reshape(FLAGS.image_size)
                        train_l_m_arr = read_file(FLAGS.L, l_train_files, index).reshape(FLAGS.image_size)
                        mask = 1.0 - np.ones(train_m_arr.shape, dtype="float32") * (train_m_arr > 0.1)
                        while True:
                            train_l_arr = read_file(FLAGS.L, l_train_files,
                                                    np.random.randint(len(l_train_files))).reshape(
                                FLAGS.image_size)
                            if np.sum(mask * train_l_arr) == 0.0: break
                            logging.info("mask and label not match !")

                        train_x_arr = read_file(FLAGS.X, l_x_train_files, index).reshape(FLAGS.image_size)
                        train_y_arr = read_file(FLAGS.Y, l_y_train_files, index).reshape(FLAGS.image_size)
                        train_z_arr = read_file(FLAGS.Z, l_z_train_files, index).reshape(FLAGS.image_size)
                        train_w_arr = read_file(FLAGS.W, l_w_train_files, index).reshape(FLAGS.image_size)

                        train_true_l.append(train_l_arr)
                        train_true_l_m.append(train_l_m_arr)
                        train_true_m.append(train_m_arr)
                        train_true_x.append(train_x_arr)
                        train_true_y.append(train_y_arr)
                        train_true_z.append(train_z_arr)
                        train_true_w.append(train_w_arr)

                        epoch = int(index / len(l_train_files))
                        index = index + 1

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    sess.run(optimizers,
                             feed_dict={
                                 l_0: np.asarray(train_true_l)[0:1, :, :, :],
                                 l_m_0: np.asarray(train_true_l_m)[0:1, :, :, :],
                                 m_0: np.asarray(train_true_m)[0:1, :, :, :],
                                 x_0: np.asarray(train_true_x)[0:1, :, :, :],
                                 y_0: np.asarray(train_true_y)[0:1, :, :, :],
                                 z_0: np.asarray(train_true_z)[0:1, :, :, :],
                                 w_0: np.asarray(train_true_w)[0:1, :, :, :],

                                 l_1: np.asarray(train_true_l)[1:2, :, :, :],
                                 l_m_1: np.asarray(train_true_l_m)[1:2, :, :, :],
                                 m_1: np.asarray(train_true_m)[1:2, :, :, :],
                                 x_1: np.asarray(train_true_x)[1:2, :, :, :],
                                 y_1: np.asarray(train_true_y)[1:2, :, :, :],
                                 z_1: np.asarray(train_true_z)[1:2, :, :, :],
                                 w_1: np.asarray(train_true_w)[1:2, :, :, :],

                                 l_2: np.asarray(train_true_l)[2:3, :, :, :],
                                 l_m_2: np.asarray(train_true_l_m)[2:3, :, :, :],
                                 m_2: np.asarray(train_true_m)[2:3, :, :, :],
                                 x_2: np.asarray(train_true_x)[2:3, :, :, :],
                                 y_2: np.asarray(train_true_y)[2:3, :, :, :],
                                 z_2: np.asarray(train_true_z)[2:3, :, :, :],
                                 w_2: np.asarray(train_true_w)[2:3, :, :, :],

                                 l_3: np.asarray(train_true_l)[3:4, :, :, :],
                                 l_m_3: np.asarray(train_true_l_m)[3:4, :, :, :],
                                 m_3: np.asarray(train_true_m)[3:4, :, :, :],
                                 x_3: np.asarray(train_true_x)[3:4, :, :, :],
                                 y_3: np.asarray(train_true_y)[3:4, :, :, :],
                                 z_3: np.asarray(train_true_z)[3:4, :, :, :],
                                 w_3: np.asarray(train_true_w)[3:4, :, :, :],
                             })
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    step += 1
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
