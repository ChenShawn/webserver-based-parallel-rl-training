import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from scipy.sparse import linalg

class cifar10Reader(object):
    dataset_dir = './cifar-10-batches-py/'
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self):
        import pickle
        file_names = ['data_batch_%d' % i for i in range(1, 6)]
        file_names = list(map(lambda x: self.dataset_dir + x, file_names))
        self.test_dict = pickle.load(open(self.dataset_dir + 'test_batch', 'rb'), encoding='bytes')
        self.dictionaries = list(map(lambda x: pickle.load(open(x, 'rb'), encoding='bytes'), file_names))
        for item in self.dictionaries:
            # The valid value is supposed to be between 0 and 1!!!
            item[b'labels'] = np.array(item[b'labels'])
            item[b'data'] = item[b'data'].astype(np.float32) / 255.0
        self.test_dict[b'data'] = self.test_dict[b'data'].astype(np.float32) / 255.0
        self.test_dict[b'labels'] = np.array(self.test_dict[b'labels'])
        print('Cifar-10 Initialization Finished!!')

    def next_batch(self, num):
        batch_idx = np.random.randint(0, 5)
        image_idx = np.random.randint(0, 10000, size=(num))
        samples = np.copy(self.dictionaries[batch_idx][b'data'][image_idx])
        labels = np.copy(self.dictionaries[batch_idx][b'labels'][image_idx])
        # Fucking bloody cifar-10 orders
        for i in range(num):
            res = samples[i, :].reshape((3,32,32)).transpose([1,2,0])
            samples[i, :] = res.reshape((1, 3072))
        samples = samples.reshape((-1, 32, 32, 3))
        return samples, labels

    def test_batch(self, begin_idx, end_idx):
        samples = np.copy(self.test_dict[b'data'][begin_idx: end_idx])
        labels = np.copy(self.test_dict[b'labels'][begin_idx: end_idx])
        for i in range(end_idx - begin_idx):
            res = samples[i, :].reshape((3,32,32)).transpose([1,2,0])
            samples[i, :] = res.reshape((1, 3072))
        samples = samples.reshape((-1, 32, 32, 3))
        return samples, labels
    
    
class mnistReader(object):
    def __init__(self):
        self.reader = input_data.read_data_sets('MNIST_data/', one_hot=True)
        
    def next_batch(self, batch_size):
        xs, ys = self.reader.train.next_batch(batch_size)
        ys = np.argmax(ys, axis=-1)
        xs = xs.reshape((-1, 28, 28, 1))
        xs_stacked = np.concatenate([xs, xs, xs], axis=-1)
        # print(xs_stacked[0].shape, xs_stacked[0].dtype, xs.max(), xs.min())
        output_xs = []
        for idx in range(xs.shape[0]):
            img = cv2.resize(xs_stacked[idx], (32, 32))
            output_xs.append(img[None, :, :, :])
        output_xs = np.concatenate(output_xs, axis=0)
        return output_xs, ys

    def test_batch(self, batch_size):
        xs, ys = self.reader.test.next_batch(batch_size)
        ys = np.argmax(ys, axis=-1)
        xs = xs.reshape((-1, 28, 28, 1))
        xs_stacked = np.concatenate([xs, xs, xs], axis=-1)
        # print(xs_stacked[0].shape, xs_stacked[0].dtype, xs.max(), xs.min())
        output_xs = []
        for idx in range(xs.shape[0]):
            img = cv2.resize(xs_stacked[idx], (32, 32))
            output_xs.append(img[None, :, :, :])
        output_xs = np.concatenate(output_xs, axis=0)
        return output_xs, ys


# ===================================================================================
cifar_reader = cifar10Reader()
mnist_reader = mnistReader()
# ====================================================================================


def build_network(input_op, output_dim=10, scope='meta', reuse=tf.AUTO_REUSE):
    params = {'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d(), 
              'bias_initializer': tf.constant_initializer(0.0)}
    net = input_op
    with tf.variable_scope(scope, reuse=reuse):
        net = tf.layers.conv2d(net, 64, 3, padding='same', activation=tf.nn.relu, name='l1', **params)
        net = tf.layers.conv2d(net, 64, 3, padding='same', activation=tf.nn.relu, name='l2', **params)
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2))
        net = tf.layers.conv2d(net, 128, 3, padding='same', activation=tf.nn.relu, name='l3', **params)
        net = tf.layers.conv2d(net, 128, 3, padding='same', activation=tf.nn.relu, name='l4', **params)
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2))
        net = tf.layers.conv2d(net, 256, 3, padding='same', activation=tf.nn.relu, name='l5', **params)
        net = tf.layers.conv2d(net, 256, 3, padding='same', activation=tf.nn.relu, name='l6', **params)
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2))
        net = tf.layers.conv2d(net, 32, 1, padding='same', activation=tf.nn.relu, name='l7', **params)
        net = tf.reshape(net, [-1, 512], name='reshaped')
        output = tf.layers.dense(net, 10, name='logits', **params)
    return output


class MetaModel(object):
    mnist_learning_rate, cifar_learning_rate, meta_learning_rate = 8e-4, 2e-4, 1e-4
    batch_size = 128
    
    def __init__(self):
        self.input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_xs')
        self.input_ys = tf.placeholder(tf.int32, [None], name='input_ys')
        self.meta = build_network(self.input_xs, scope='meta', reuse=False)
        self.mnist_adapt = build_network(self.input_xs, scope='mnist', reuse=False)
        self.cifar_adapt = build_network(self.input_xs, scope='cifar', reuse=False)
        
        meta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='meta')
        mnist_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist')
        cifar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cifar')
        meta_vars.sort(key=lambda x: x.name)
        mnist_vars.sort(key=lambda x: x.name)
        cifar_vars.sort(key=lambda x: x.name)
        
        with tf.name_scope('losses'):
            self.assign_mnist = [x.assign(y) for x, y in zip(meta_vars, mnist_vars)]
            self.assign_cifar = [x.assign(y) for x, y in zip(meta_vars, cifar_vars)]
            
            mnist_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_ys, logits=self.mnist_adapt)
            cifar_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_ys, logits=self.cifar_adapt)
            mnist_loss, cifar_loss = tf.reduce_mean(mnist_loss), tf.reduce_mean(cifar_loss)
            mnist_trainer = tf.train.AdamOptimizer(self.mnist_learning_rate)
            cifar_trainer = tf.train.AdamOptimizer(self.cifar_learning_rate)
            meta_trainer = tf.train.AdamOptimizer(self.meta_learning_rate)
            
            mnist_grads_and_vars = mnist_trainer.compute_gradients(mnist_loss, var_list=mnist_vars)
            cifar_grads_and_vars = cifar_trainer.compute_gradients(cifar_loss, var_list=cifar_vars)
            meta_grads_and_vars = []
            for mgv, cgv, mv in zip(mnist_grads_and_vars, cifar_grads_and_vars, meta_vars):
                gradient_sum_for_meta = mgv[0] + cgv[0]
                meta_grads_and_vars.append((gradient_sum_for_meta, mv))
            self.mnist_optim = mnist_trainer.apply_gradients(mnist_grads_and_vars)
            self.cifar_optim = cifar_trainer.apply_gradients(cifar_grads_and_vars)
            self.meta_optim = meta_trainer.apply_gradients(meta_grads_and_vars)

            self.mnist_map = tf.reduce_mean(tf.cast(tf.equal(self.input_ys, tf.argmax(self.mnist_adapt, axis=-1, output_type=tf.int32)), tf.float32))
            self.cifar_map = tf.reduce_mean(tf.cast(tf.equal(self.input_ys, tf.argmax(self.cifar_adapt, axis=-1, output_type=tf.int32)), tf.float32))

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.InteractiveSession(config=config)
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] Build meta model')
        
    def single_step_update(self):
        self.sess.run(self.assign_mnist)
        self.sess.run(self.assign_cifar)
        mnist_xs, mnist_ys = mnist_reader.next_batch(self.batch_size)
        cifar_xs, cifar_ys = cifar_reader.next_batch(self.batch_size)
        self.sess.run(self.mnist_optim, feed_dict={self.input_xs: mnist_xs, self.input_ys: mnist_ys})
        self.sess.run(self.cifar_optim, feed_dict={self.input_xs: cifar_xs, self.input_ys: cifar_ys})
        meta_xs = np.concatenate([mnist_xs, cifar_xs], axis=0)
        meta_ys = np.concatenate([mnist_ys, cifar_ys], axis=0)
        self.sess.run(self.meta_optim, feed_dict={self.input_xs: meta_xs, self.input_ys: meta_ys})

    def evaluate_mnist(self):
        mapval_list = []
        for it in range(100):
            mnist_xs, mnist_ys = mnist_reader.test_batch(100)
            mapval = self.sess.run(self.mnist_map, feed_dict={self.input_xs: mnist_xs, self.input_ys: mnist_ys})
            mapval_list.append(mapval)
        return np.array(mapval_list, dtype=np.float32).mean()

    def evaluate_cifar(self):
        mapval_list = []
        for it in range(100):
            cifar_xs, cifar_ys = cifar_reader.test_batch(it * 64, (it + 1) * 64)
            mapval = self.sess.run(self.mnist_map, feed_dict={self.input_xs: cifar_xs, self.input_ys: cifar_ys})
            mapval_list.append(mapval)
        return np.array(mapval_list, dtype=np.float32).mean()
        
    def load_state(self, model_path, saver=None):
        import re
        print(" [*] Reading checkpoints in {}...".format(model_path))
        if saver is None:
            saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(model_path, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint in {}".format(model_path))
            return False, 0
        
    def save_state(self, fname, saver=None):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if saver is None:
            saver = tf.train.Saver()
        saver.save(self.sess, fname + '/model')
        return saver


if __name__ == '__main__':
    # xs, ys = mnist_reader.test_batch(1)
    # xss, yss = cifar_reader.test_batch(100, 101)

    # print(xs.shape, xs.dtype, ys.shape, ys.dtype, xs.min(), xs.max())
    # print(xss.shape, xss.dtype, yss.shape, yss.dtype, xss.min(), xss.max())

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(xs[0])
    # plt.axis('off')
    # plt.subplot(122)
    # plt.imshow(xss[0])
    # plt.axis('off')
    # plt.show()


    meta_model = MetaModel()
    MAX_TRAIN_ITER = 20000
    BATCH_SIZE = 256

    mnist_summaries, cifar_summaries = [], []
    x_steps = list(range(0, MAX_TRAIN_ITER, 100))
    for idx in range(MAX_TRAIN_ITER):
        meta_model.single_step_update()
        if idx % 100 == 0:
            mnist_xs, mnist_ys = mnist_reader.test_batch(BATCH_SIZE)
            cifar_xs, cifar_ys = cifar_reader.test_batch((idx // 100) * BATCH_SIZE, (idx // 100 + 1) * BATCH_SIZE)
            mnist_map = meta_model.sess.run(meta_model.mnist_map, feed_dict={meta_model.input_xs: mnist_xs, meta_model.input_ys: mnist_ys})
            cifar_map = meta_model.sess.run(meta_model.cifar_map, feed_dict={meta_model.input_xs: cifar_xs, meta_model.input_ys: cifar_ys})
            mnist_summaries.append(mnist_map)
            cifar_summaries.append(cifar_map)
            print('MNIST: {}, CIFAR-10: {}'.format(mnist_map, cifar_map))
    meta_model.save_state()

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(x_steps, mnist_summaries, linewidth=1.5)
    plt.plot(x_steps, cifar_summaries, linewidth=1.5)
    plt.legend(['MNIST', 'CIFAR'], loc='best')
    plt.xlabel('Training iterations')
    plt.ylabel('Precision on test set')
    plt.savefig('./images/meta_training.png')
    plt.show()
