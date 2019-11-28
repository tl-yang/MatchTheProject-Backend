# top 1 accuracy 0.9249791286257038 top k accuracy 0.9747623788455786
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.ops import control_flow_ops

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 16002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(os.path.dirname(__file__), 'tf_checkpoint'),
                           'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '../data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '../data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('tf_log_dir', './tf_log_tf', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'inference', 'Running mode. One of {"train", "valid", "test"}')

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
FLAGS = tf.app.flags.FLAGS



class CnnRecognizer(object):
    def __init__(self):
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options={'allow_growth': True}))
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        self.graph = self.build_graph(top_k=3)
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(self.sess, ckpt)

    def build_graph(self, top_k):
        keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
        labels = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
        is_training = tf.compat.v1.placeholder(dtype=tf.bool, shape=[], name='train_flag')
        # with tf.device('/gpu:0'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                          scope='fc2')
        loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=logits, axis=1), labels), tf.float32))

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.compat.v1.get_variable("step", [], initializer=tf.compat.v1.constant_initializer(0.0),
                                                trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits)

        # tf.summary.scalar('loss', loss)
        # tf.summary.scalar('accuracy', accuracy)
        # merged_summary_op = tf.summary.merge_all()
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(
            input_tensor=tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=labels, k=top_k), tf.float32))

        return {'images': images,
                'labels': labels,
                'keep_prob': keep_prob,
                'top_k': top_k,
                'global_step': global_step,
                'train_op': train_op,
                'loss': loss,
                'is_training': is_training,
                'accuracy': accuracy,
                'accuracy_top_k': accuracy_in_top_k,
                # 'merged_summary_op': merged_summary_op,
                'predicted_distribution': probabilities,
                'predicted_index_top_k': predicted_index_top_k,
                'predicted_val_top_k': predicted_val_top_k}


    def produce(self, image):
        temp_image = np.asarray(image) / 255.0
        temp_image = temp_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
        predict_val, predict_index = self.sess.run(
            [self.graph['predicted_val_top_k'], self.graph['predicted_index_top_k']],
            feed_dict={self.graph['images']: temp_image,
                       self.graph['keep_prob']: 1.0,
                       self.graph['is_training']: False})
        return predict_index, predict_val


def main():
    rec = CnnRecognizer()
    image_path = os.path.join(os.path.dirname(__file__), 'log', 'inverted_level1_1574695845.3789456.jpg')
    temp_image = Image.open(image_path).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    st = time.time()
    final_predict_index, final_predict_val = rec.produce(temp_image)
    print(time.time() - st)
    # final_predict_val, final_predict_index = inference(image_path)
    logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                     final_predict_val))


if __name__ == "__main__":
    main()
    # tf.compat.v1.app.run()
