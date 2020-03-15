# _*_ coding:utf-8 _*_
import tensorflow as tf
from model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK
import math
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 1')
tf.flags.DEFINE_list('image_size', [512, 512, 1], 'image size, default: [155,240,240]')
tf.flags.DEFINE_float('learning_rate', 2e-5, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/train/X', 'X files for training')
tf.flags.DEFINE_string('L', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/train/L', 'Y files for training')
tf.flags.DEFINE_string('F', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/train/NEW_F', 'Y files for training')
tf.flags.DEFINE_string('M', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/train/M', 'Y files for training')
tf.flags.DEFINE_string('X_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/test/X', 'X files for training')
tf.flags.DEFINE_string('L_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/test/L', 'Y files for training')
tf.flags.DEFINE_string('F_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/test/NEW_F', 'Y files for training')
tf.flags.DEFINE_string('M_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/chest_xray/test/M', 'Y files for training')

tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 1000, 'default: 100')
tf.flags.DEFINE_float('display_epoch', 1, 'default: 1')
tf.flags.DEFINE_integer('epoch_steps', 1341, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('stage', "train", 'default: train')

tf.flags.DEFINE_string('load_lp_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


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


def mynorm(input):
    if (np.max(input) - np.min(input)) != 0:
        output = (input - np.min(input)
                  ) / (np.max(input) - np.min(input))
    else:
        # print("单值图：",np.max(input))
        output = (input - input)  # +1.0
    return output


def save_images(image_dirct, checkpoints_dir, file_index=""):
    for key in image_dirct:
        save_image(np.asarray(image_dirct[key])[0, :, :, :], key + "_" + file_index,
                   dir=checkpoints_dir + "/samples", form=".tiff")


def save_image(image, name, dir="./samples", form=".tiff"):
    try:
        os.makedirs(dir)
    except os.error:
        pass
    if image.shape[2] == 1:
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image[:, :, 0]), dir + "/" + name + form)
    else:
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image), dir + "/" + name + ".mha")


def read_file(l_path, Label_train_files, index, out_size=None, inpu_form="", out_form="", norm=True):
    train_range = len(Label_train_files)
    file_name = l_path + "/" + Label_train_files[index % train_range].replace(inpu_form, out_form)
    L_img = SimpleITK.ReadImage(file_name)
    L_arr = SimpleITK.GetArrayFromImage(L_img)

    if len(L_arr.shape) == 2:
        img = cv2.merge([L_arr[:, :], L_arr[:, :], L_arr[:, :]])
    elif L_arr.shape[2] == 1:
        img = cv2.merge([L_arr[:, :, 0], L_arr[:, :, 0], L_arr[:, :, 0]])
    elif L_arr.shape[2] == 3:
        img = cv2.merge([L_arr[:, :, 0], L_arr[:, :, 1], L_arr[:, :, 2]])
    if out_size == None:
        img = cv2.resize(img, (FLAGS.image_size[0], FLAGS.image_size[1]), interpolation=cv2.INTER_NEAREST)
        img = np.asarray(img)[:, :, 0:FLAGS.image_size[2]].astype('float32')
    else:
        img = cv2.resize(img, (out_size[0], out_size[1]), interpolation=cv2.INTER_NEAREST)
        img = np.asarray(img)[:, :, 0:out_size[2]].astype('float32')
    if norm == True:
        img = mynorm(img)
    return img


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

        graph = tf.get_default_graph()
        gan = GAN(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.ngf)
        input_shape = [int(FLAGS.batch_size / 4), FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]]
        G_optimizer, D_optimizer = gan.optimize()

        G_grad_list = []
        D_grad_list = []
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device("/gpu:0"):
                with tf.name_scope("GPU_0"):
                    l_0 = tf.placeholder(tf.float32, shape=input_shape)
                    x_0 = tf.placeholder(tf.float32, shape=input_shape)
                    f_0 = tf.placeholder(tf.float32, shape=input_shape)
                    m_0 = tf.placeholder(tf.float32, shape=input_shape)

                    loss_list_0, image_list_0 = gan.model(l_0, f_0, m_0, x_0)
                    j_list_0 = gan.judge_list
                    tensor_name_dirct_0 = gan.tenaor_name
                    variables_list_0 = gan.get_variables()
                    G_grad_0 = G_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                    D_grad_0 = D_optimizer.compute_gradients(loss_list_0[1], var_list=variables_list_0[1])
                    G_grad_list.append(G_grad_0)
                    D_grad_list.append(D_grad_0)
            with tf.device("/gpu:1"):
                with tf.name_scope("GPU_1"):
                    l_1 = tf.placeholder(tf.float32, shape=input_shape)
                    x_1 = tf.placeholder(tf.float32, shape=input_shape)
                    f_1 = tf.placeholder(tf.float32, shape=input_shape)
                    m_1 = tf.placeholder(tf.float32, shape=input_shape)

                    loss_list_1, image_list_1 = gan.model(l_1, f_1, m_1, x_1)
                    variables_list_1 = gan.get_variables()
                    G_grad_1 = G_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                    D_grad_1 = D_optimizer.compute_gradients(loss_list_1[1], var_list=variables_list_1[1])
                    G_grad_list.append(G_grad_1)
                    D_grad_list.append(D_grad_1)
            with tf.device("/gpu:2"):
                with tf.name_scope("GPU_2"):
                    l_2 = tf.placeholder(tf.float32, shape=input_shape)
                    x_2 = tf.placeholder(tf.float32, shape=input_shape)
                    f_2 = tf.placeholder(tf.float32, shape=input_shape)
                    m_2 = tf.placeholder(tf.float32, shape=input_shape)

                    loss_list_2, image_list_2 = gan.model(l_2, f_2, m_2, x_2)
                    variables_list_2 = gan.get_variables()
                    G_grad_2 = G_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                    D_grad_2 = D_optimizer.compute_gradients(loss_list_2[1], var_list=variables_list_2[1])
                    G_grad_list.append(G_grad_2)
                    D_grad_list.append(D_grad_2)
            with tf.device("/gpu:3"):
                with tf.name_scope("GPU_3"):
                    l_3 = tf.placeholder(tf.float32, shape=input_shape)
                    x_3 = tf.placeholder(tf.float32, shape=input_shape)
                    f_3 = tf.placeholder(tf.float32, shape=input_shape)
                    m_3 = tf.placeholder(tf.float32, shape=input_shape)

                    loss_list_3, image_list_3 = gan.model(l_3, f_3, m_3, x_3)
                    variables_list_3 = gan.get_variables()
                    G_grad_3 = G_optimizer.compute_gradients(loss_list_3[0], var_list=variables_list_3[0])
                    D_grad_3 = D_optimizer.compute_gradients(loss_list_3[1], var_list=variables_list_3[1])
                    G_grad_list.append(G_grad_3)
                    D_grad_list.append(D_grad_3)

        G_ave_grad = average_gradients(G_grad_list)
        D_ave_grad = average_gradients(D_grad_list)
        G_optimizer_op = G_optimizer.apply_gradients(G_ave_grad)
        D_optimizer_op = D_optimizer.apply_gradients(D_ave_grad)
        optimizers = [G_optimizer_op, D_optimizer_op]

        gan.image_summary(image_list_0)
        gan.histogram_summary(j_list_0)
        image_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'image')])
        loss_list_summary = tf.placeholder(tf.float32)
        gan.loss_summary(loss_list_summary)
        summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')])
        train_writer = tf.summary.FileWriter(checkpoints_dir + "/train", graph)
        val_writer = tf.summary.FileWriter(checkpoints_dir + "/val", graph)
        saver = tf.train.Saver()
        variables_list = gan.get_variables()

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                           gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            sess.run(tf.global_variables_initializer())
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

            if FLAGS.load_lp_model is not None:
                seg_latest_checkpoint = tf.train.latest_checkpoint("checkpoints/" + FLAGS.load_lp_model)
                seg_saver = tf.train.Saver(variables_list[2])
                seg_saver.restore(sess, seg_latest_checkpoint)

            sess.graph.finalize()
            logging.info("start step:" + str(step))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                logging.info("tensor_name_dirct:\n" + str(tensor_name_dirct_0))
                f_train_files = read_filename(FLAGS.F)
                # x_train_files = read_filename(FLAGS.X)
                index = 0
                epoch = 0
                train_loss_list = []
                while not coord.should_stop() and epoch <= FLAGS.epoch:

                    train_true_l = []
                    train_true_f = []
                    train_true_m = []
                    train_true_x = []
                    for b in range(FLAGS.batch_size):
                        train_m_arr = read_file(FLAGS.M, f_train_files, index, inpu_form=".jpeg", out_form=".tiff")
                        train_f_arr = read_file(FLAGS.F, f_train_files, index, inpu_form=".jpeg", out_form=".tiff")
                        train_l_arr = read_file(FLAGS.L, f_train_files, index, inpu_form=".jpeg", out_form=".tiff",
                                                norm=False)
                        train_x_arr = read_file(FLAGS.X, f_train_files, index, inpu_form=".tiff", out_form=".jpeg")

                        train_true_l.append(train_l_arr)
                        train_true_f.append(train_f_arr)
                        train_true_m.append(train_m_arr)
                        train_true_x.append(train_x_arr)

                        epoch = int(index / len(f_train_files))
                        index = index + 1

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    _, train_image_summary_op, train_losses = sess.run(
                        [optimizers, image_summary_op, loss_list_0],
                        feed_dict={
                            l_0: np.asarray(train_true_l)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                            f_0: np.asarray(train_true_f)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                            m_0: np.asarray(train_true_m)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                            x_0: np.asarray(train_true_x)[0:1 * int(FLAGS.batch_size / 4), :, :, :],

                            l_1: np.asarray(train_true_l)[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            f_1: np.asarray(train_true_f)[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            m_1: np.asarray(train_true_m)[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            x_1: np.asarray(train_true_x)[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4),
                                 :, :, :],

                            l_2: np.asarray(train_true_l)[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            f_2: np.asarray(train_true_f)[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            m_2: np.asarray(train_true_m)[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            x_2: np.asarray(train_true_x)[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4),
                                 :, :, :],

                            l_3: np.asarray(train_true_l)[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            f_3: np.asarray(train_true_f)[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            m_3: np.asarray(train_true_m)[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                            x_3: np.asarray(train_true_x)[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4),
                                 :, :, :],
                        })
                    train_loss_list.append(train_losses)
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    if step == 0 or step % int(FLAGS.epoch_steps / 2 - 1) == 0 or step == int(
                            FLAGS.epoch_steps * FLAGS.epoch / 4):
                        logging.info('-----------Train summary start-------------')
                        train_summary_op = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(train_loss_list)})
                        train_writer.add_summary(train_image_summary_op, step)
                        train_writer.add_summary(train_summary_op, step)
                        train_writer.flush()
                        train_loss_list = []
                        logging.info('-----------Train summary end-------------')

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                        logging.info(
                            "-----------val epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                        val_loss_list = []
                        val_index = 0

                        f_val_files = read_filename(FLAGS.F_test)
                        # x_val_files = read_filename(FLAGS.X_test)
                        for j in range(int(math.ceil(len(f_val_files) / FLAGS.batch_size))):
                            val_true_l = []
                            val_true_f = []
                            val_true_m = []
                            val_true_x = []
                            for b in range(FLAGS.batch_size):
                                val_m_arr = read_file(FLAGS.M_test, f_val_files, val_index, inpu_form=".jpeg",
                                                      out_form=".tiff")
                                val_f_arr = read_file(FLAGS.F_test, f_val_files, val_index, inpu_form=".jpeg",
                                                      out_form=".tiff")
                                val_l_arr = read_file(FLAGS.L_test, f_val_files, val_index, inpu_form=".jpeg",
                                                      out_form=".tiff", norm=False)
                                val_x_arr = read_file(FLAGS.X_test, f_val_files, val_index, inpu_form=".tiff",
                                                      out_form=".jpeg")

                                val_true_l.append(val_l_arr)
                                val_true_f.append(val_f_arr)
                                val_true_m.append(val_m_arr)
                                val_true_x.append(val_x_arr)
                                val_index += 1

                            val_losses_0, \
                            val_losses_1, \
                            val_losses_2, \
                            val_losses_3, \
                            val_image_summary_op, val_image_list_0, val_image_list_1, val_image_list_2, val_image_list_3 = sess.run(
                                [loss_list_0,
                                 loss_list_1,
                                 loss_list_2,
                                 loss_list_3,
                                 image_summary_op, image_list_0, image_list_1, image_list_2, image_list_3],
                                feed_dict={
                                    l_0: np.asarray(val_true_l)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                                    f_0: np.asarray(val_true_f)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                                    m_0: np.asarray(val_true_m)[0:1 * int(FLAGS.batch_size / 4), :, :, :],
                                    x_0: np.asarray(val_true_x)[0:1 * int(FLAGS.batch_size / 4), :, :, :],

                                    l_1: np.asarray(val_true_l)[
                                         1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4), :, :, :],
                                    f_1: np.asarray(val_true_f)[
                                         1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4), :, :, :],
                                    m_1: np.asarray(val_true_m)[
                                         1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4), :, :, :],
                                    x_1: np.asarray(val_true_x)[
                                         1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4), :, :, :],

                                    l_2: np.asarray(val_true_l)[
                                         2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4), :, :, :],
                                    f_2: np.asarray(val_true_f)[
                                         2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4), :, :, :],
                                    m_2: np.asarray(val_true_m)[
                                         2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4), :, :, :],
                                    x_2: np.asarray(val_true_x)[
                                         2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4), :, :, :],

                                    l_3: np.asarray(val_true_l)[
                                         3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4), :, :, :],
                                    f_3: np.asarray(val_true_f)[
                                         3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4), :, :, :],
                                    m_3: np.asarray(val_true_m)[
                                         3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4), :, :, :],
                                    x_3: np.asarray(val_true_x)[
                                         3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4), :, :, :],
                                })
                            val_loss_list.append(val_losses_0)
                            val_loss_list.append(val_losses_1)
                            val_loss_list.append(val_losses_2)
                            val_loss_list.append(val_losses_3)

                            if j % 1 == 0:
                                save_images(val_image_list_0, checkpoints_dir, str(j) + "_0")
                                save_images(val_image_list_1, checkpoints_dir, str(j) + "_1")
                                save_images(val_image_list_2, checkpoints_dir, str(j) + "_2")
                                save_images(val_image_list_3, checkpoints_dir, str(j) + "_3")

                        val_summary_op = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(val_loss_list)})
                        val_writer.add_summary(val_image_summary_op, step)
                        val_writer.add_summary(val_summary_op, step)
                        val_writer.flush()

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
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
