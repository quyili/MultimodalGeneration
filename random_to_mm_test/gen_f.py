# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_string('load_model', "20190704-1239",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_string('tensor_name', "GPU_0/Reshape_3:0", "default: None")
tf.flags.DEFINE_integer('epoch_steps', 15070, '463 or 5480, default: 5480')
tf.flags.DEFINE_integer('epochs', 1, '463 or 5480, default: 5480')


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
        os.makedirs("./jpg")
        os.makedirs("./tiff")
    except os.error:
        pass

    graph = tf.get_default_graph()
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    print(meta_graph_path)
    saver = tf.train.import_meta_graph(meta_graph_path)

    # for op in tf.get_default_graph().get_tensor_by_name():
    #     print(op.name)
    #

    f_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.tensor_name)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        index = 0
        while index <= FLAGS.epoch_steps * FLAGS.epochs:
            print("image gen start:" + str(index))
            f = sess.run(f_rm)
            full_x = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                     np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
            cv2.imwrite("./jpg/fake_f_"+  str(index)  +".jpg", full_x)
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(f)[0, :, :, 0]),
                                 "./tiff/fake_f_" + str(index) + ".tiff")
            print("image gen end:" + str(index))

            index += 1


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
