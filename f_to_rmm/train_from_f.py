# _*_ coding:utf-8 _*_
import tensorflow as tf
from unsupervision_mode import GAN
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
tf.flags.DEFINE_string('X', '../mydata/BRATS2015/trainT1', 'X files for training')
tf.flags.DEFINE_string('Y', '../mydata/BRATS2015/trainT2', 'Y files for training')
tf.flags.DEFINE_string('Z', '../mydata/BRATS2015/trainT1c', 'X files for training')
tf.flags.DEFINE_string('W', '../mydata/BRATS2015/trainFlair', 'Y files for training')
tf.flags.DEFINE_string('L', '../mydata/BRATS2015/trainLabelV', 'Y files for training')
tf.flags.DEFINE_string('X_test', '../mydata/BRATS2015/testT1', 'X files for training')
tf.flags.DEFINE_string('Y_test', '../mydata/BRATS2015/testT2', 'Y files for training')
tf.flags.DEFINE_string('Z_test', '../mydata/BRATS2015/testT1c', 'X files for training')
tf.flags.DEFINE_string('W_test', '../mydata/BRATS2015/testFlair', 'Y files for training')
tf.flags.DEFINE_string('L_test', '../mydata/BRATS2015/testLabelV', 'Y files for training')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 50, 'default: 100')
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


def save_images(image_list, checkpoints_dir, file_index):
    true_x, true_y, true_f, true_l,\
    gen_x_g, gen_y_g, gen_x_g_t, gen_y_g_t, gen_l_g, gen_l_g_by_x, gen_l_g_by_y,gen_f_x_g_r, gen_f_y_g_r,\
    trans_x_r, trans_y_r, trans_x_t, trans_y_t, trans_x_c_r, trans_y_c_r, trans_l_f_by_x, trans_l_f_by_y = image_list
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(true_x)[0, :, :, 0]),
                         checkpoints_dir + "/samples/true_x_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(true_y)[0, :, :, 0]),
                         checkpoints_dir + "/samples/true_y_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(true_f)[0, :, :, 0]),
                         checkpoints_dir + "/samples/true_f_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(true_l)[0, :, :, 0]),
                         checkpoints_dir + "/samples/true_l_" + str(file_index) + ".tiff")

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_x_g)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_x_g_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_y_g)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_y_g_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_x_g_t)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_x_g_t_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_y_g_t)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_y_g_t_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_l_g)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_l_g_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_l_g_by_x)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_l_g_by_x_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_l_g_by_y)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_l_g_by_y_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(gen_f_x_g_r)[0, :, :, 0]),
                         checkpoints_dir + "/samples/gen_f_x_g_r_" + str(file_index) + ".tiff")

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_x_r)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_x_r_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_y_r)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_y_r_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_x_t)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_x_t_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_y_t)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_y_t_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_x_c_r)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_x_c_r_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_y_c_r)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_y_c_r_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_l_f_by_x)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_l_f_by_x_" + str(file_index) + ".tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(trans_l_f_by_y)[0, :, :, 0]),
                         checkpoints_dir + "/samples/trans_l_f_by_y_" + str(file_index) + ".tiff")


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
            G_optimizer, D_optimizer = gan.optimize()

            G_grad_list = []
            D_grad_list = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/gpu:0"):
                    with tf.name_scope("GPU_0"):
                        x_0 = tf.placeholder(tf.float32, shape=input_shape)
                        y_0 = tf.placeholder(tf.float32, shape=input_shape)
                        label_expand_0 = tf.placeholder(tf.float32,shape=input_shape)
                        z_0 = tf.placeholder(tf.float32, shape=input_shape)
                        w_0 = tf.placeholder(tf.float32, shape=input_shape)
                        rande_f_0 = tf.placeholder(tf.int32)
                        rande_train_0 = tf.placeholder(tf.int32)
                        loss_list_0 = gan.run(x_0, y_0, z_0, w_0,label_expand_0,rande_f_0,rande_train_0)
                        image_list_0, code_list_0, j_list_0=gan.image_list, gan.code_list, gan.judge_list
                        evaluation_list_0 = gan.evaluation(gan.image_list)
                        evaluation_code_list_0 = gan.evaluation_code(code_list_0)
                        variables_list_0 = gan.get_variables()
                        G_grad_0 = G_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                        D_grad_0 = D_optimizer.compute_gradients(loss_list_0[1], var_list=variables_list_0[1])
                        G_grad_list.append(G_grad_0)
                        D_grad_list.append(D_grad_0)
                with tf.device("/gpu:1"):
                    with tf.name_scope("GPU_1"):
                        x_1 = tf.placeholder(tf.float32, shape=input_shape)
                        y_1 = tf.placeholder(tf.float32, shape=input_shape)
                        label_expand_1 = tf.placeholder(tf.float32, shape=input_shape)
                        z_1 = tf.placeholder(tf.float32, shape=input_shape)
                        w_1 = tf.placeholder(tf.float32, shape=input_shape)
                        rande_f_1 = tf.placeholder(tf.int32)
                        rande_train_1 = tf.placeholder(tf.int32)
                        loss_list_1 = gan.run(x_1, y_1, z_1, w_1, label_expand_1, rande_f_1, rande_train_1)
                        image_list_1, code_list_1, j_list_1 = gan.image_list, gan.code_list, gan.judge_list
                        evaluation_list_1 = gan.evaluation(image_list_1)
                        evaluation_code_list_1 = gan.evaluation_code(code_list_1)
                        variables_list_1 = gan.get_variables()
                        G_grad_1 = G_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                        D_grad_1 = D_optimizer.compute_gradients(loss_list_1[1], var_list=variables_list_1[1])
                        G_grad_list.append(G_grad_1)
                        D_grad_list.append(D_grad_1)
                with tf.device("/gpu:2"):
                    with tf.name_scope("GPU_2"):
                        x_2 = tf.placeholder(tf.float32, shape=input_shape)
                        y_2 = tf.placeholder(tf.float32, shape=input_shape)
                        label_expand_2 = tf.placeholder(tf.float32,shape=input_shape)
                        z_2 = tf.placeholder(tf.float32, shape=input_shape)
                        w_2 = tf.placeholder(tf.float32, shape=input_shape)
                        rande_f_2 = tf.placeholder(tf.int32)
                        rande_train_2 = tf.placeholder(tf.int32)
                        loss_list_2 = gan.run(x_2, y_2, z_2, w_2, label_expand_2, rande_f_2, rande_train_2)
                        image_list_2, code_list_2, j_list_2 = gan.image_list, gan.code_list, gan.judge_list
                        evaluation_list_2 = gan.evaluation(image_list_2)
                        evaluation_code_list_2 = gan.evaluation_code(code_list_2)
                        variables_list_2 = gan.get_variables()
                        G_grad_2 = G_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                        D_grad_2 = D_optimizer.compute_gradients(loss_list_2[1], var_list=variables_list_2[1])
                        G_grad_list.append(G_grad_2)
                        D_grad_list.append(D_grad_2)
                with tf.device("/gpu:3"):
                    with tf.name_scope("GPU_3"):
                        x_3 = tf.placeholder(tf.float32, shape=input_shape)
                        y_3 = tf.placeholder(tf.float32, shape=input_shape)
                        label_expand_3 = tf.placeholder(tf.float32,shape=input_shape)
                        z_3 = tf.placeholder(tf.float32, shape=input_shape)
                        w_3 = tf.placeholder(tf.float32, shape=input_shape)
                        rande_f_3 = tf.placeholder(tf.int32)
                        rande_train_3 = tf.placeholder(tf.int32)
                        loss_list_3 = gan.run(x_3, y_3, z_3, w_3, label_expand_3, rande_f_3, rande_train_3)
                        image_list_3, code_list_3, j_list_3 = gan.image_list, gan.code_list, gan.judge_list
                        evaluation_list_3 = gan.evaluation(image_list_3)
                        evaluation_code_list_3 = gan.evaluation_code(code_list_3)
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
            image_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'image'),
                                                 tf.get_collection(tf.GraphKeys.SUMMARIES, 'discriminator')])

            loss_list_summary = tf.placeholder(tf.float32)
            evaluation_list_summary = tf.placeholder(tf.float32)
            evaluation_code_list_summary = tf.placeholder(tf.float32)

            gan.loss_summary(loss_list_summary)
            gan.evaluation_summary(evaluation_list_summary)
            gan.evaluation_code_summary(evaluation_code_list_summary)

            summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'evaluation'),
                                           tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')])
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
                index = 0
                epoch = 0
                train_loss_list = []
                train_evaluation_list = []
                train_evaluation_code_list = []
                while not coord.should_stop() and epoch <= FLAGS.epoch:
                    train_true_x = []
                    train_true_y = []
                    train_true_z = []
                    train_true_w = []
                    train_true_l = []
                    for b in range(FLAGS.batch_size):
                        train_L_arr_ = read_file(FLAGS.L, Label_train_files, index)
                        train_X_arr_ = read_file(FLAGS.X, Label_train_files, index)
                        train_Y_arr_ = read_file(FLAGS.Y, Label_train_files, index)
                        train_Z_arr_ = read_file(FLAGS.Z, Label_train_files, index)
                        train_W_arr_ = read_file(FLAGS.W, Label_train_files, index)
                        L_arr = np.asarray(train_L_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))*5
                        X_arr = np.asarray(train_X_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        Y_arr = np.asarray(train_Y_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        Z_arr = np.asarray(train_Z_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                        W_arr = np.asarray(train_W_arr_).reshape(
                            (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))

                        train_true_x.append(X_arr)
                        train_true_y.append(Y_arr)
                        train_true_z.append(Z_arr)
                        train_true_w.append(W_arr)
                        train_true_l.append(L_arr)
                        epoch = int(index / len(Label_train_files))
                        index = index + 1

                    # rande_f = np.random.randint(4)
                    # rande_train: np.random.randint(6)

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    _, train_image_summary_op, train_losses, train_evaluations, train_evaluation_codes = sess.run(
                        [optimizers, image_summary_op, loss_list_0, evaluation_list_0, evaluation_code_list_0],
                        feed_dict={
                            x_0: np.asarray(train_true_x)[0:1, :, :, :],
                            y_0: np.asarray(train_true_y)[0:1, :, :, :],
                            z_0: np.asarray(train_true_z)[0:1, :, :, :],
                            w_0: np.asarray(train_true_w)[0:1, :, :, :],
                            rande_f_0:np.random.randint(4),
                            rande_train_0: np.random.randint(6),
                            label_expand_0: np.asarray(train_true_l)[0:1, :, :, :],

                            x_1: np.asarray(train_true_x)[1:2, :, :, :],
                            y_1: np.asarray(train_true_y)[1:2, :, :, :],
                            z_1: np.asarray(train_true_z)[1:2, :, :, :],
                            w_1: np.asarray(train_true_w)[1:2, :, :, :],
                            rande_f_1: np.random.randint(4),
                            rande_train_1: np.random.randint(6),
                            label_expand_1: np.asarray(train_true_l)[1:2, :, :, :],

                            x_2: np.asarray(train_true_x)[2:3, :, :, :],
                            y_2: np.asarray(train_true_y)[2:3, :, :, :],
                            z_2: np.asarray(train_true_z)[2:3, :, :, :],
                            w_2: np.asarray(train_true_w)[2:3, :, :, :],
                            rande_f_2: np.random.randint(4),
                            rande_train_2: np.random.randint(6),
                            label_expand_2: np.asarray(train_true_l)[2:3, :, :, :],

                            x_3: np.asarray(train_true_x)[3:4, :, :, :],
                            y_3: np.asarray(train_true_y)[3:4, :, :, :],
                            z_3: np.asarray(train_true_z)[3:4, :, :, :],
                            w_3: np.asarray(train_true_w)[3:4, :, :, :],
                            rande_f_3: np.random.randint(4),
                            rande_train_3: np.random.randint(6),
                            label_expand_3: np.asarray(train_true_l)[3:4, :, :, :],
                        })
                    train_loss_list.append(train_losses)
                    train_evaluation_list.append(train_evaluations)
                    train_evaluation_code_list.append(train_evaluation_codes)
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    if step == 0 or step % int(FLAGS.epoch_steps / 2 - 1) == 0 or step == int(
                            FLAGS.epoch_steps * FLAGS.epoch / 4):
                        logging.info('-----------Train summary start-------------')
                        train_summary_op = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(train_loss_list),
                                       evaluation_list_summary: mean_list(train_evaluation_list),
                                       evaluation_code_list_summary: mean_list(train_evaluation_code_list)})
                        train_writer.add_summary(train_image_summary_op, step)
                        train_writer.add_summary(train_summary_op, step)
                        train_writer.flush()
                        logging.info('-----------Train summary end-------------')

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                        logging.info(
                            "-----------val epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                        val_loss_list = []
                        val_evaluation_list = []
                        val_evaluation_code_list = []
                        val_index = 0
                        Label_test_files = read_filename(FLAGS.L_test)
                        for j in range(int(math.ceil(len(Label_test_files) / FLAGS.batch_size))):
                            val_true_x = []
                            val_true_y = []
                            val_true_z = []
                            val_true_w = []
                            val_true_l = []
                            for b in range(FLAGS.batch_size):
                                val_L_arr_ = read_file(FLAGS.L, Label_test_files, val_index)
                                val_X_arr_ = read_file(FLAGS.X, Label_test_files, val_index)
                                val_Y_arr_ = read_file(FLAGS.Y, Label_test_files, val_index)
                                val_Z_arr_ = read_file(FLAGS.Z, Label_test_files, val_index)
                                val_W_arr_ = read_file(FLAGS.W, Label_test_files, val_index)
                                L_arr = np.asarray(val_L_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2])) * 5
                                X_arr = np.asarray(val_X_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                Y_arr = np.asarray(val_Y_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                Z_arr = np.asarray(val_Z_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))
                                W_arr = np.asarray(val_W_arr_).reshape(
                                    (FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]))

                                val_true_x.append(X_arr)
                                val_true_y.append(Y_arr)
                                val_true_z.append(Z_arr)
                                val_true_w.append(W_arr)
                                val_true_l.append(L_arr)
                                val_index += 1

                            val_losses_0, val_evaluations_0, val_evaluation_codes_0, \
                            val_losses_1, val_evaluations_1, val_evaluation_codes_1, \
                            val_losses_2, val_evaluations_2, val_evaluation_codes_2, \
                            val_losses_3, val_evaluations_3, val_evaluation_codes_3, \
                            val_image_summary_op, val_image_list_0, val_image_list_1, val_image_list_2, val_image_list_3 = sess.run(
                                [loss_list_0, evaluation_list_0, evaluation_code_list_0,
                                 loss_list_1, evaluation_list_1, evaluation_code_list_1,
                                 loss_list_2, evaluation_list_2, evaluation_code_list_2,
                                 loss_list_3, evaluation_list_3, evaluation_code_list_3,
                                 image_summary_op, image_list_0, image_list_1, image_list_2, image_list_3,],
                                feed_dict={
                                    x_0: np.asarray(val_true_x)[0:1, :, :, :],
                                    y_0: np.asarray(val_true_y)[0:1, :, :, :],
                                    z_0: np.asarray(val_true_z)[0:1, :, :, :],
                                    w_0: np.asarray(val_true_w)[0:1, :, :, :],
                                    rande_f_0: np.random.randint(4),
                                    rande_train_0: np.random.randint(6),
                                    label_expand_0: np.asarray(val_true_l)[0:1, :, :, :],

                                    x_1: np.asarray(val_true_x)[1:2, :, :, :],
                                    y_1: np.asarray(val_true_y)[1:2, :, :, :],
                                    z_1: np.asarray(val_true_z)[1:2, :, :, :],
                                    w_1: np.asarray(val_true_w)[1:2, :, :, :],
                                    rande_f_1: np.random.randint(4),
                                    rande_train_1: np.random.randint(6),
                                    label_expand_1: np.asarray(val_true_l)[1:2, :, :, :],

                                    x_2: np.asarray(val_true_x)[2:3, :, :, :],
                                    y_2: np.asarray(val_true_y)[2:3, :, :, :],
                                    z_2: np.asarray(val_true_z)[2:3, :, :, :],
                                    w_2: np.asarray(val_true_w)[2:3, :, :, :],
                                    rande_f_2: np.random.randint(4),
                                    rande_train_2: np.random.randint(6),
                                    label_expand_2: np.asarray(val_true_l)[2:3, :, :, :],

                                    x_3: np.asarray(val_true_x)[3:4, :, :, :],
                                    y_3: np.asarray(val_true_y)[3:4, :, :, :],
                                    z_3: np.asarray(val_true_z)[3:4, :, :, :],
                                    w_3: np.asarray(val_true_w)[3:4, :, :, :],
                                    rande_f_3: np.random.randint(4),
                                    rande_train_3: np.random.randint(6),
                                    label_expand_3: np.asarray(val_true_l)[3:4, :, :, :],
                                })
                            val_loss_list.append(val_losses_0)
                            val_loss_list.append(val_losses_1)
                            val_loss_list.append(val_losses_2)
                            val_loss_list.append(val_losses_3)
                            val_evaluation_list.append(val_evaluations_0)
                            val_evaluation_list.append(val_evaluations_1)
                            val_evaluation_list.append(val_evaluations_2)
                            val_evaluation_list.append(val_evaluations_3)
                            val_evaluation_code_list.append(val_evaluation_codes_0)
                            val_evaluation_code_list.append(val_evaluation_codes_1)
                            val_evaluation_code_list.append(val_evaluation_codes_2)
                            val_evaluation_code_list.append(val_evaluation_codes_3)

                            if j == 0:
                                save_images(val_image_list_3, checkpoints_dir, val_index - 1)

                        val_summary_op = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(val_loss_list),
                                       evaluation_list_summary: mean_list(val_evaluation_list),
                                       evaluation_code_list_summary: mean_list(val_evaluation_code_list)})
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
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
