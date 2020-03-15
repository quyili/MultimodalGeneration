# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK
import cv2
import math
from scipy.stats import norm

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_string('load_model', "20190719-1738",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_string('code_tensor_name', "GPU_0/random_normal_1:0", "default: None")
tf.flags.DEFINE_string('f_tensor_name', "GPU_0/Reshape_4:0", "default: None")
tf.flags.DEFINE_string('m_tensor_name', "GPU_0/Reshape_5:0", "default: None")
tf.flags.DEFINE_string('j_f_tensor_name', "GPU_3/D_F_1/conv5/conv5/BiasAdd:0", "default: None")
tf.flags.DEFINE_integer('epoch_steps', 20, ' default: 15070')
tf.flags.DEFINE_integer('epochs', 1, ' default: 1')
tf.flags.DEFINE_float('max_count', 50, 'default: 50')
tf.flags.DEFINE_float('mae', 0.05, 'default: 0.05')


def get_mask_from_f(imgfile):
    # imgfile = "full_x.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    c_list = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = c_list[-2], c_list[-1]
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    # savefile="mask.tiff"
    return np.asarray(1.0 - img / 255.0, dtype="float32")


def train():
    if FLAGS.load_model is not None:
        if FLAGS.savefile is not None:
            checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")

    else:
        print("<load_model> is None.")
        return
    try:
        os.makedirs("./N_F")
        os.makedirs("./N_F/Temp")
    except os.error:
        pass
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_path)

    graph = tf.get_default_graph()
    code_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.code_tensor_name)
    f_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.f_tensor_name)
    mask_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.m_tensor_name)
    j_f_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.j_f_tensor_name)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        index = 0
        while index <= FLAGS.epoch_steps * FLAGS.epochs:
            print("image gen start:" + str(index))

            count = 0
            best_mae = 1000.0
            while True:
                code1 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
                f, m = sess.run([f_rm, mask_rm], feed_dict={code_rm: code1.reshape((1, 4096))})
                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg", jpg_f)
                m_arr_1 = get_mask_from_f("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')
                mae = np.mean(np.abs(m_arr_1 - m_arr_2))
                # 根据结构完整度过滤
                if mae <= FLAGS.mae:
                    break
                elif mae < best_mae:
                    best_mae = mae
                    best_f = f
                    best_m = m
                # 根据生成次数过滤
                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break
                count = count + 1

            count = 0
            best_mae = 1000.0
            while True:
                code2 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
                f, m = sess.run([f_rm, mask_rm], feed_dict={code_rm: code2.reshape((1, 4096))})
                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg", jpg_f)
                m_arr_1 = get_mask_from_f("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')
                mae = np.mean(np.abs(m_arr_1 - m_arr_2))
                # 根据结构完整度过滤
                if mae <= FLAGS.mae:
                    break
                elif mae < best_mae:
                    best_mae = mae
                    best_f = f
                    best_m = m
                # 根据生成次数过滤
                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break
                count = count + 1

            count = 0
            best_mae = 1000.0
            while True:
                code3 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
                f, m = sess.run([f_rm, mask_rm], feed_dict={code_rm: code3.reshape((1, 4096))})
                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg", jpg_f)
                m_arr_1 = get_mask_from_f("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')
                mae = np.mean(np.abs(m_arr_1 - m_arr_2))
                # 根据结构完整度过滤
                if mae <= FLAGS.mae:
                    break
                elif mae < best_mae:
                    best_mae = mae
                    best_f = f
                    best_m = m
                # 根据生成次数过滤
                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break
                count = count + 1

            count = 0
            best_mae = 1000.0
            while True:
                code4 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
                f, m = sess.run([f_rm, mask_rm], feed_dict={code_rm: code4.reshape((1, 4096))})
                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg", jpg_f)
                m_arr_1 = get_mask_from_f("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')
                mae = np.mean(np.abs(m_arr_1 - m_arr_2))
                # 根据结构完整度过滤
                if mae <= FLAGS.mae:
                    break
                elif mae < best_mae:
                    best_mae = mae
                    best_f = f
                    best_m = m
                # 根据生成次数过滤
                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break
                count = count + 1

            count = 0
            best_mae = 1000.0
            while True:
                code5 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
                f, m = sess.run([f_rm, mask_rm], feed_dict={code_rm: code5.reshape((1, 4096))})
                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg", jpg_f)
                m_arr_1 = get_mask_from_f("./N_F/Temp/f_" + str(index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')
                mae = np.mean(np.abs(m_arr_1 - m_arr_2))
                # 根据结构完整度过滤
                if mae <= FLAGS.mae:
                    break
                elif mae < best_mae:
                    best_mae = mae
                    best_f = f
                    best_m = m
                # 根据生成次数过滤
                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break
                count = count + 1

            # code2 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
            # code3 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
            # code4 = np.random.normal(0.0, 1.0, (4096)).astype('float32')
            # code5 = np.random.normal(0.0, 1.0, (4096)).astype('float32')

            codes = np.zeros((9, 9, 4096))
            codes[0, 0, :] = code1[:]
            codes[0, 8, :] = code2[:]
            codes[8, 0, :] = code3[:]
            codes[8, 8, :] = code4[:]
            codes[4, 4, :] = code5[:]

            code_lines12 = code_line(code1, code2, 7)
            for i in range(7):
                codes[0, i + 1, :] = code_lines12[i]
            code_lines13 = code_line(code1, code3, 7)
            for i in range(7):
                codes[i + 1, 0, :] = code_lines13[i]
            code_lines24 = code_line(code2, code4, 7)
            for i in range(7):
                codes[i + 1, 8, :] = code_lines24[i]
            code_lines34 = code_line(code3, code4, 7)
            for i in range(7):
                codes[8, i + 1, :] = code_lines34[i]

            code_lines512 = code_line(code5, codes[0, 4, :], 3)
            for i in range(3):
                codes[4 - (i + 1), 4, :] = code_lines512[i]
            code_lines513 = code_line(code5, codes[4, 0, :], 3)
            for i in range(3):
                codes[4, 4 - (i + 1), :] = code_lines513[i]
            code_lines524 = code_line(code5, codes[4, 8, :], 3)
            for i in range(3):
                codes[4, 4 + (i + 1), :] = code_lines524[i]
            code_lines534 = code_line(code5, codes[8, 4, :], 3)
            for i in range(3):
                codes[4 + (i + 1), 4, :] = code_lines534[i]

            code_lines51 = code_line(code5, code1, 3)
            for i in range(3):
                codes[4 - (i + 1), 4 - (i + 1), :] = code_lines51[i]
            code_lines52 = code_line(code5, code2, 3)
            for i in range(3):
                codes[4 - (i + 1), 4 + (i + 1), :] = code_lines52[i]
            code_lines53 = code_line(code5, code3, 3)
            for i in range(3):
                codes[4 + (i + 1), 4 - (i + 1), :] = code_lines53[i]
            code_lines54 = code_line(code5, code4, 3)
            for i in range(3):
                codes[4 + (i + 1), 4 + (i + 1), :] = code_lines54[i]

            code_lines1_12_5 = code_line(codes[3, 4, :], codes[0, 1, :], 2)
            codes[1, 3, :] = code_line(codes[2, 4, :], codes[0, 2, :], 1)[0]
            for i in range(2):
                codes[4 - (i + 2), 4 - (i + 1), :] = code_lines1_12_5[i]
            code_lines1_13_5 = code_line(codes[4, 3, :], codes[0, 1, :], 2)
            codes[3, 1, :] = code_line(codes[4, 2, :], codes[2, 0, :], 1)[0]
            for i in range(2):
                codes[4 - (i + 1), 4 - (i + 2), :] = code_lines1_13_5[i]
            code_lines2_12_5 = code_line(codes[3, 4, :], codes[0, 7, :], 2)
            codes[1, 5, :] = code_line(codes[2, 4, :], codes[0, 6, :], 1)[0]
            for i in range(2):
                codes[4 - (i + 2), 4 + (i + 1), :] = code_lines2_12_5[i]
            code_lines2_24_5 = code_line(codes[4, 5, :], codes[1, 8, :], 2)
            codes[3, 7, :] = code_line(codes[4, 6, :], codes[2, 8, :], 1)[0]
            for i in range(2):
                codes[4 - (i + 1), 4 + (i + 2), :] = code_lines2_24_5[i]
            code_lines3_13_5 = code_line(codes[4, 3, :], codes[7, 0, :], 2)
            codes[5, 1, :] = code_line(codes[4, 2, :], codes[7, 0, :], 1)[0]
            for i in range(2):
                codes[4 + (i + 1), 4 - (i + 2), :] = code_lines3_13_5[i]
            code_lines3_34_5 = code_line(codes[5, 4, :], codes[8, 1, :], 2)
            codes[7, 3, :] = code_line(codes[6, 4, :], codes[8, 2, :], 1)[0]
            for i in range(2):
                codes[4 + (i + 2), 4 - (i + 1), :] = code_lines3_34_5[i]
            code_lines4_34_5 = code_line(codes[5, 4, :], codes[8, 7, :], 2)
            codes[7, 5, :] = code_line(codes[6, 4, :], codes[8, 6, :], 1)[0]
            for i in range(2):
                codes[4 + (i + 2), 4 + (i + 1), :] = code_lines4_34_5[i]
            code_lines4_24_5 = code_line(codes[4, 5, :], codes[7, 8, :], 2)
            codes[5, 7, :] = code_line(codes[4, 6, :], codes[6, 8, :], 1)[0]
            for i in range(2):
                codes[4 + (i + 1), 4 + (i + 2), :] = code_lines4_24_5[i]

            figure = np.zeros((184 * 9, 144 * 9))
            for i in range(9):
                for j in range(9):
                    z_sample = codes[i, j, :]
                    z_sample = z_sample.reshape((1, 4096))
                    f, m, j_f = sess.run([f_rm, mask_rm, j_f_rm], feed_dict={code_rm: z_sample})
                    figure[i * 184: (i + 1) * 184, j * 144: (j + 1) * 144] = np.asarray(f)[0, :, :, 0]
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(figure.astype('float32')),
                                 "./N_F/" + str(index) + ".tiff")

            print("image gen end:" + str(index))
            index += 1


def code_line(code1, code2, N, L=4096):
    code_line = []
    code = np.zeros((L))
    code[:] = code1[:]
    for i in range(N):
        for j in range(L):
            if j % N == i:
                code[j] = code2[j]
        code_temp = np.zeros((L))
        code_temp[:] = code[:]
        code_line.append(code_temp)
    return code_line


def main(unused_argv):
    train()
    os.system("rm -r " + "./N_F/Temp")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
