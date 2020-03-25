# _*_ coding:utf-8 _*_
import tensorflow as tf
from lesion_model import GAN
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
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_integer('ngf', 1, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', 'E:/project/MultimodalGeneration/data/TC19/test/X', 'X files for training')
tf.flags.DEFINE_string('L', 'E:/project/MultimodalGeneration/data/TC19/test/L', 'Y files for training')
tf.flags.DEFINE_string('X_test', 'E:/project/MultimodalGeneration/data/TC19/test/X', 'X files for training')
tf.flags.DEFINE_string('L_test', 'E:/project/MultimodalGeneration/data/TC19/test/L', 'Y files for training')
# tf.flags.DEFINE_string('X', '/GPUFS/nsccgz_ywang_1/quyili/DATA/TC19/train/X', 'X files for training')
# tf.flags.DEFINE_string('L', '/GPUFS/nsccgz_ywang_1/quyili/DATA/TC19/train/L', 'Y files for training')
# tf.flags.DEFINE_string('X_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/TC19/test/X', 'X files for training')
# tf.flags.DEFINE_string('L_test', '/GPUFS/nsccgz_ywang_1/quyili/DATA/TC19/test/L', 'Y files for training')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 1, 'default: 100')
tf.flags.DEFINE_float('display_epoch', 1, 'default: 1')
tf.flags.DEFINE_integer('epoch_steps', 8, '463 or 5480, default: 5480')
tf.flags.DEFINE_string('stage', "train", 'default: train')

# 每个特征图单元的default box数量
default_box_size = [4, 6, 6, 6, 4, 4]
# 最小default box面积比例
min_box_scale = 0.05
# 最大default box面积比例
max_box_scale = 0.9
# 每个特征层的面积比例
# numpy生成等差数组，效果等同于论文中的s_k=s_min+(s_max-s_min)*(k-1)/(m-1)
default_box_scale = np.linspace(min_box_scale, max_box_scale, num=np.amax(default_box_size))
print('##   default_box_scale:' + str(default_box_scale))
# default box 尺寸长宽比例
box_aspect_ratio = [
    [1.0, 1.25, 2.0, 3.0],
    [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
    [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
    [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
    [1.0, 1.25, 2.0, 3.0],
    [1.0, 1.25, 2.0, 3.0]
]
classes_size = 5
feature_maps_shape = [[64, 64, ], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
# Jaccard相似度判断阀值
jaccard_value = 0.6
# 背景分类的值
background_classes_val = 4


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
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output


def read_file(l_path, Label_train_files, index, out_size=None, inpu_form="", out_form="", norm=False):
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


# 检测数据是否正常
def check_numerics(input_dataset, message):
    if str(input_dataset).find('Tensor') == 0:
        input_dataset = tf.check_numerics(input_dataset, message)
    else:
        dataset = np.array(input_dataset)
        nan_count = np.count_nonzero(dataset != dataset)
        inf_count = len(dataset[dataset == float("inf")])
        n_inf_count = len(dataset[dataset == float("-inf")])
        if nan_count > 0 or inf_count > 0 or n_inf_count > 0:
            data_error = '【' + message + '】出现数据错误！【nan：' + str(nan_count) + '|inf：' + str(inf_count) + '|-inf：' + str(
                n_inf_count) + '】'
            raise Exception(data_error)
    return input_dataset


# 初始化、整理训练数据
def generate_all_default_boxs():
    # 全部按照比例计算并生成一张图像产生的每个default box的坐标以及长宽
    # 用于后续的jaccard匹配
    all_default_boxes = []
    for index, map_shape in zip(range(len(feature_maps_shape)), feature_maps_shape):
        width = int(map_shape[0])
        height = int(map_shape[1])
        cell_scale = default_box_scale[index]
        for x in range(width):
            for y in range(height):
                for ratio in box_aspect_ratio[index]:
                    center_x = (x / float(width)) + (0.5 / float(width))
                    center_y = (y / float(height)) + (0.5 / float(height))
                    box_width = np.sqrt(cell_scale * ratio)
                    box_height = np.sqrt(cell_scale / ratio)
                    all_default_boxes.append([center_x, center_y, box_width, box_height])
    all_default_boxes = np.array(all_default_boxes)
    # 检查数据是否正确
    all_default_boxes = check_numerics(all_default_boxes, 'all_default_boxes')
    return all_default_boxes


# 整理生成groundtruth数据
all_default_boxs = generate_all_default_boxs()
all_default_boxs_len = len(all_default_boxs)


def generate_groundtruth_data(input_actual_data):
    # 生成空数组，用于保存groundtruth
    input_actual_data_len = len(input_actual_data)
    gt_class = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_location = np.zeros((input_actual_data_len, all_default_boxs_len, 4))
    gt_positives_jacc = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_positives = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_negatives = np.zeros((input_actual_data_len, all_default_boxs_len))
    background_jacc = max(0, (jaccard_value - 0.2))
    # 初始化正例训练数据
    for img_index in range(input_actual_data_len):
        for pre_actual in input_actual_data[img_index]:
            gt_class_val = pre_actual[-1:][0]  # todo
            gt_box_val = pre_actual[:-1]
            for boxe_index in range(all_default_boxs_len):
                jacc = jaccard(gt_box_val, all_default_boxs[boxe_index])
                if jacc > jaccard_value or jacc == jaccard_value:
                    gt_class[img_index][boxe_index] = gt_class_val
                    gt_location[img_index][boxe_index] = gt_box_val
                    gt_positives_jacc[img_index][boxe_index] = jacc
                    gt_positives[img_index][boxe_index] = 1
                    gt_negatives[img_index][boxe_index] = 0
        # 如果没有正例，则随机创建一个正例，预防nan
        if np.sum(gt_positives[img_index]) == 0:
            # print('【没有匹配jacc】:'+str(input_actual_data[img_index]))
            random_pos_index = np.random.randint(low=0, high=all_default_boxs_len, size=1)[0]
            gt_class[img_index][random_pos_index] = background_classes_val
            gt_location[img_index][random_pos_index] = [0, 0, 0, 0]
            gt_positives_jacc[img_index][random_pos_index] = jaccard_value
            gt_positives[img_index][random_pos_index] = 1
            gt_negatives[img_index][random_pos_index] = 0
        # 正负例比值 1:3
        gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
        if (gt_neg_end_count + np.sum(gt_positives[img_index])) > all_default_boxs_len:
            gt_neg_end_count = all_default_boxs_len - np.sum(gt_positives[img_index])
        # 随机选择负例
        gt_neg_index = np.random.randint(low=0, high=all_default_boxs_len, size=gt_neg_end_count)
        for r_index in gt_neg_index:
            if gt_positives_jacc[img_index][r_index] < background_jacc:
                gt_class[img_index][r_index] = background_classes_val
                gt_positives[img_index][r_index] = 0
                gt_negatives[img_index][r_index] = 1
    return gt_class, gt_location, gt_positives, gt_negatives


# jaccard算法
# 计算IOU，rect1、rect2格式为[center_x,center_y,width,height]
def jaccard(rect1, rect2):
    x_overlap = max(0, (min(rect1[0] + (rect1[2] / 2), rect2[0] + (rect2[2] / 2)) - max(rect1[0] - (rect1[2] / 2),
                                                                                        rect2[0] - (rect2[2] / 2))))
    y_overlap = max(0, (min(rect1[1] + (rect1[3] / 2), rect2[1] + (rect2[3] / 2)) - max(rect1[1] - (rect1[3] / 2),
                                                                                        rect2[1] - (rect2[3] / 2))))
    intersection = x_overlap * y_overlap
    # 删除超出图像大小的部分
    rect1_width_sub = 0
    rect1_height_sub = 0
    rect2_width_sub = 0
    rect2_height_sub = 0
    if (rect1[0] - rect1[2] / 2) < 0: rect1_width_sub += 0 - (rect1[0] - rect1[2] / 2)
    if (rect1[0] + rect1[2] / 2) > 1: rect1_width_sub += (rect1[0] + rect1[2] / 2) - 1
    if (rect1[1] - rect1[3] / 2) < 0: rect1_height_sub += 0 - (rect1[1] - rect1[3] / 2)
    if (rect1[1] + rect1[3] / 2) > 1: rect1_height_sub += (rect1[1] + rect1[3] / 2) - 1
    if (rect2[0] - rect2[2] / 2) < 0: rect2_width_sub += 0 - (rect2[0] - rect2[2] / 2)
    if (rect2[0] + rect2[2] / 2) > 1: rect2_width_sub += (rect2[0] + rect2[2] / 2) - 1
    if (rect2[1] - rect2[3] / 2) < 0: rect2_height_sub += 0 - (rect2[1] - rect2[3] / 2)
    if (rect2[1] + rect2[3] / 2) > 1: rect2_height_sub += (rect2[1] + rect2[3] / 2) - 1
    area_box_a = (rect1[2] - rect1_width_sub) * (rect1[3] - rect1_height_sub)
    area_box_b = (rect2[2] - rect2_width_sub) * (rect2[3] - rect2_height_sub)
    union = area_box_a + area_box_b - intersection
    if intersection > 0 and union > 0:
        return intersection / union
    else:
        return 0


def read_txt_file(l_path, Label_train_files, index, inpu_form=".mha"):
    actual_item = []
    train_range = len(Label_train_files)
    file_name = l_path + "/" + Label_train_files[index % train_range].replace(inpu_form, ".txt")
    with open(file_name) as f:
        for line in f.readlines():
            line = line.replace("\n", "").split(" ")
            actual_item.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), int(line[0])])
    return actual_item


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
        gan = GAN(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, classes_size, FLAGS.ngf)
        input_shape = [int(FLAGS.batch_size / 4), FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]]
        G_optimizer = gan.optimize()

        G_grad_list = []
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device("/gpu:0"):
                with tf.name_scope("GPU_0"):
                    X_0 = tf.placeholder(tf.float32, shape=input_shape, name='input_image')
                    GT_class_0 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                dtype=tf.int32,
                                                name='groundtruth_class')
                    GT_location_0 = tf.placeholder(shape=[None, all_default_boxs_len, 4],
                                                   dtype=tf.float32,
                                                   name='groundtruth_location')
                    GT_positives_0 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_positives')
                    GT_negatives_0 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_negatives')
                    loss_list_0, feature_class_0, feature_location_0 = gan.model(X_0,
                                                                                 GT_class_0,
                                                                                 GT_location_0,
                                                                                 GT_positives_0,
                                                                                 GT_negatives_0)
                    feature_class_softmax_0, box_top_index_0, box_top_value_0 = gan.pred(
                        classes_size, feature_class_0,
                        background_classes_val,
                        all_default_boxs_len)
                    tensor_name_dirct_0 = gan.tenaor_name
                    variables_list_0 = gan.get_variables()
                    G_grad_0 = G_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                    G_grad_list.append(G_grad_0)
            with tf.device("/gpu:1"):
                with tf.name_scope("GPU_1"):
                    X_1 = tf.placeholder(tf.float32, shape=input_shape, name='input_image')
                    GT_class_1 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                dtype=tf.int32,
                                                name='groundtruth_class')
                    GT_location_1 = tf.placeholder(shape=[None, all_default_boxs_len, 4],
                                                   dtype=tf.float32,
                                                   name='groundtruth_location')
                    GT_positives_1 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_positives')
                    GT_negatives_1 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_negatives')
                    loss_list_1, feature_class_1, feature_location_1 = gan.model(X_1,
                                                                                 GT_class_1,
                                                                                 GT_location_1,
                                                                                 GT_positives_1,
                                                                                 GT_negatives_1)
                    variables_list_1 = gan.get_variables()
                    G_grad_1 = G_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                    G_grad_list.append(G_grad_1)
            with tf.device("/gpu:2"):
                with tf.name_scope("GPU_2"):
                    X_2 = tf.placeholder(tf.float32, shape=input_shape, name='input_image')
                    GT_class_2 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                dtype=tf.int32,
                                                name='groundtruth_class')
                    GT_location_2 = tf.placeholder(shape=[None, all_default_boxs_len, 4],
                                                   dtype=tf.float32,
                                                   name='groundtruth_location')
                    GT_positives_2 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_positives')
                    GT_negatives_2 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_negatives')
                    loss_list_2, feature_class_2, feature_location_2 = gan.model(X_2,
                                                                                 GT_class_2,
                                                                                 GT_location_2,
                                                                                 GT_positives_2,
                                                                                 GT_negatives_2)
                    variables_list_2 = gan.get_variables()
                    G_grad_2 = G_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                    G_grad_list.append(G_grad_2)
            with tf.device("/gpu:3"):
                with tf.name_scope("GPU_3"):
                    X_3 = tf.placeholder(tf.float32, shape=input_shape, name='input_image')
                    GT_class_3 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                dtype=tf.int32,
                                                name='groundtruth_class')
                    GT_location_3 = tf.placeholder(shape=[None, all_default_boxs_len, 4],
                                                   dtype=tf.float32,
                                                   name='groundtruth_location')
                    GT_positives_3 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_positives')
                    GT_negatives_3 = tf.placeholder(shape=[None, all_default_boxs_len],
                                                    dtype=tf.float32,
                                                    name='groundtruth_negatives')
                    loss_list_3, feature_class_3, feature_location_3 = gan.model(X_3,
                                                                                 GT_class_3,
                                                                                 GT_location_3,
                                                                                 GT_positives_3,
                                                                                 GT_negatives_3)
                    variables_list_3 = gan.get_variables()
                    G_grad_3 = G_optimizer.compute_gradients(loss_list_3[0], var_list=variables_list_3[0])
                    G_grad_list.append(G_grad_3)

        G_ave_grad = average_gradients(G_grad_list)
        G_optimizer_op = G_optimizer.apply_gradients(G_ave_grad)
        optimizers = [G_optimizer_op]

        loss_list_summary = tf.placeholder(tf.float32)

        gan.loss_summary(loss_list_summary)
        summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')])
        train_writer = tf.summary.FileWriter(checkpoints_dir + "/train", graph)
        val_writer = tf.summary.FileWriter(checkpoints_dir + "/val", graph)
        saver = tf.train.Saver()

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
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

            sess.graph.finalize()
            logging.info("start step:" + str(step))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                logging.info("tensor_name_dirct:\n" + str(tensor_name_dirct_0))
                l_train_files = read_filename(FLAGS.L)
                index = 0
                epoch = 0
                train_loss_list = []
                while not coord.should_stop() and epoch <= FLAGS.epoch:

                    train_true_l = []
                    train_true_x = []
                    for b in range(FLAGS.batch_size):
                        train_l_arr = read_txt_file(FLAGS.L, l_train_files, index)
                        train_x_arr = read_file(FLAGS.X, l_train_files, index, out_size=[512, 512, 3],
                                                inpu_form=".txt", out_form=".mha")
                        train_true_l.append(train_l_arr)
                        train_true_x.append(train_x_arr[:, :, 1:2])
                        epoch = int(index / len(l_train_files))
                        index = index + 1

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    gt_class_0, gt_location_0, gt_positives_0, gt_negatives_0 = generate_groundtruth_data(
                        train_true_l[0 * int(FLAGS.batch_size / 4):1 * int(FLAGS.batch_size / 4)])
                    gt_class_1, gt_location_1, gt_positives_1, gt_negatives_1 = generate_groundtruth_data(
                        train_true_l[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4)])
                    gt_class_2, gt_location_2, gt_positives_2, gt_negatives_2 = generate_groundtruth_data(
                        train_true_l[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4)])
                    gt_class_3, gt_location_3, gt_positives_3, gt_negatives_3 = generate_groundtruth_data(
                        train_true_l[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4)])

                    _, train_losses = sess.run(
                        [optimizers, loss_list_0], feed_dict={
                            X_0: train_true_x[0 * int(FLAGS.batch_size / 4):1 * int(FLAGS.batch_size / 4)],
                            GT_class_0: gt_class_0,
                            GT_location_0: gt_location_0,
                            GT_positives_0: gt_positives_0,
                            GT_negatives_0: gt_negatives_0,
                            X_1: train_true_x[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4)],
                            GT_class_1: gt_class_1,
                            GT_location_1: gt_location_1,
                            GT_positives_1: gt_positives_1,
                            GT_negatives_1: gt_negatives_1,
                            X_2: train_true_x[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4)],
                            GT_class_2: gt_class_2,
                            GT_location_2: gt_location_2,
                            GT_positives_2: gt_positives_2,
                            GT_negatives_2: gt_negatives_2,
                            X_3: train_true_x[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4)],
                            GT_class_3: gt_class_3,
                            GT_location_3: gt_location_3,
                            GT_positives_3: gt_positives_3,
                            GT_negatives_3: gt_negatives_3
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

                        l_val_files = read_filename(FLAGS.L_test)
                        for j in range(int(math.ceil(len(l_val_files) / FLAGS.batch_size))):
                            val_true_l = []
                            val_true_x = []
                            for b in range(FLAGS.batch_size):
                                val_l_arr = read_txt_file(FLAGS.L_test, l_val_files, val_index)
                                val_x_arr = read_file(FLAGS.X_test, l_val_files, val_index, out_size=[512, 512, 3],
                                                      inpu_form=".txt", out_form=".mha")
                                logging.info(l_val_files[val_index % len(l_val_files)])
                                val_true_l.append(val_l_arr)
                                val_true_x.append(val_x_arr[:, :, 1:2])
                                val_index += 1

                            gt_class_0, gt_location_0, gt_positives_0, gt_negatives_0 = generate_groundtruth_data(
                                val_true_l[0 * int(FLAGS.batch_size / 4):1 * int(FLAGS.batch_size / 4)])
                            gt_class_1, gt_location_1, gt_positives_1, gt_negatives_1 = generate_groundtruth_data(
                                val_true_l[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4)])
                            gt_class_2, gt_location_2, gt_positives_2, gt_negatives_2 = generate_groundtruth_data(
                                val_true_l[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4)])
                            gt_class_3, gt_location_3, gt_positives_3, gt_negatives_3 = generate_groundtruth_data(
                                val_true_l[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4)])

                            val_losses_0, \
                            val_losses_1, \
                            val_losses_2, \
                            val_losses_3, \
                            f_class, f_location, f_class_softmax, box_top_index, box_top_value = sess.run([
                                loss_list_0, loss_list_1, loss_list_2, loss_list_3,
                                feature_class_0, feature_location_0, feature_class_softmax_0, box_top_index_0,
                                box_top_value_0], feed_dict={
                                X_0: val_true_x[0 * int(FLAGS.batch_size / 4):1 * int(FLAGS.batch_size / 4)],
                                GT_class_0: gt_class_0,
                                GT_location_0: gt_location_0,
                                GT_positives_0: gt_positives_0,
                                GT_negatives_0: gt_negatives_0,
                                X_1: val_true_x[1 * int(FLAGS.batch_size / 4):2 * int(FLAGS.batch_size / 4)],
                                GT_class_1: gt_class_1,
                                GT_location_1: gt_location_1,
                                GT_positives_1: gt_positives_1,
                                GT_negatives_1: gt_negatives_1,
                                X_2: val_true_x[2 * int(FLAGS.batch_size / 4):3 * int(FLAGS.batch_size / 4)],
                                GT_class_2: gt_class_2,
                                GT_location_2: gt_location_2,
                                GT_positives_2: gt_positives_2,
                                GT_negatives_2: gt_negatives_2,
                                X_3: val_true_x[3 * int(FLAGS.batch_size / 4):4 * int(FLAGS.batch_size / 4)],
                                GT_class_3: gt_class_3,
                                GT_location_3: gt_location_3,
                                GT_positives_3: gt_positives_3,
                                GT_negatives_3: gt_negatives_3,
                            })

                            val_loss_list.append(val_losses_0)
                            val_loss_list.append(val_losses_1)
                            val_loss_list.append(val_losses_2)
                            val_loss_list.append(val_losses_3)

                            # print("IN:", gt_class_0.shape, gt_location_0.shape, gt_positives_0.shape,
                            #       gt_negatives_0.shape, )
                            # print("OUT:", f_class.shape, f_location.shape)
                            # top_shape = np.shape(box_top_index)
                            # pred_class = []
                            # pred_class_val = []
                            # pred_location = []
                            # for i in range(top_shape[0]):
                            #     item_img_class = []
                            #     item_img_class_val = []
                            #     item_img_location = []
                            #     for j in range(top_shape[1]):
                            #         p_class_val = f_class_softmax[i][box_top_index[i][j]]
                            #         if p_class_val < 0.5:
                            #             continue
                            #         p_class = np.argmax(f_class[i][box_top_index[i][j]])
                            #         if p_class == background_classes_val:
                            #             continue
                            #         p_location = f_location[i][box_top_index[i][j]]
                            #         if p_location[0] < 0 or p_location[1] < 0 or p_location[2] < 0 or p_location[
                            #             3] < 0 or p_location[
                            #             2] == 0 or p_location[3] == 0:
                            #             continue
                            #         is_box_filter = False
                            #         for f_index in range(len(item_img_class)):
                            #             if jaccard(p_location, item_img_location[f_index]) > 0.3 and p_class == \
                            #                     item_img_class[
                            #                         f_index]:
                            #                 is_box_filter = True
                            #                 break
                            #         if is_box_filter == False:
                            #             item_img_class.append(p_class)
                            #             item_img_class_val.append(p_class_val)
                            #             item_img_location.append(p_location)
                            #     pred_class.append(item_img_class)
                            #     pred_class_val.append(item_img_class_val)
                            #     pred_location.append(item_img_location)
                            # print('pred_class:' + str(pred_class))
                            # print('pred_class_val:' + str(pred_class_val))
                            # print('pred_location:' + str(pred_location))
                            # print('Labels:' + str(
                            #     val_true_l[0 * int(FLAGS.batch_size / 4):1 * int(FLAGS.batch_size / 4)]))

                        val_summary_op = sess.run(
                            summary_op,
                            feed_dict={loss_list_summary: mean_list(val_loss_list)})
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
