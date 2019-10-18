import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from scipy.sparse import linalg

# ================================================================================
EPSILON = 0.05
MAX_TRAIN_ITER = 5000
BATCH_SIZE = 256
MNIST_LR, CIFAR_LR, META_LR = 8e-4, 2e-4, 1e-4
ADV_MNIST_LR, ADV_CIFAR_LR = 8e-5, 2e-5
META_SAVE_DIR = './meta_train/'
ADV_SAVE_DIR = './adv_train/'
TEST_MODE = True
# ================================================================================

class cifar10Reader(object):
    # dataset_dir = '/home/zcx/Documents/datasets/cifar-10-batches-py/'
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


def fgsm_op(input_op, loss_op, epsilon=0.05):
    """fgsm_op
    FGSM implementation using tensorflow builtin static graph
    return: an op representing the adversarial examples generated using FGSM
    """
    adv_perturb = tf.gradients(loss_op, input_op)[0]
    adv_perturb = tf.reshape(adv_perturb, [-1, 32 * 32 * 3])
    norm_adv_perturb = tf.sign(tf.nn.l2_normalize(adv_perturb, axis=-1))
    norm_adv_perturb = tf.reshape(norm_adv_perturb, [-1, 32, 32, 3])
    return tf.stop_gradient(input_op + epsilon * norm_adv_perturb)


class MetaModelBase(object):
    sess = None

    def __init__(self):
        self.input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_xs')
        self.mnist_input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3], name='mnist_input_xs')
        self.mnist_input_ys = tf.placeholder(tf.int32, [None], name='mnist_input_ys')
        self.cifar_input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3], name='cifar_input_xs')
        self.cifar_input_ys = tf.placeholder(tf.int32, [None], name='cifar_input_ys')

    def single_step_update(self):
        raise NotImplementedError

    def evaluate_mnist(self):
        mapval_list = []
        for it in range(100):
            mnist_xs, mnist_ys = mnist_reader.test_batch(100)
            mapval = self.sess.run(self.mnist_map, feed_dict={self.mnist_input_xs: mnist_xs, self.mnist_input_ys: mnist_ys})
            mapval_list.append(mapval)
        return np.array(mapval_list, dtype=np.float32).mean()

    def evaluate_cifar(self):
        mapval_list = []
        for it in range(100):
            cifar_xs, cifar_ys = cifar_reader.test_batch(it * 64, (it + 1) * 64)
            mapval = self.sess.run(self.cifar_map, feed_dict={self.cifar_input_xs: cifar_xs, self.cifar_input_ys: cifar_ys})
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


class MetaModel(MetaModelBase):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.meta = build_network(self.input_xs, scope='meta', reuse=False)
        self.mnist_adapt = build_network(self.mnist_input_xs, scope='mnist', reuse=False)
        self.cifar_adapt = build_network(self.cifar_input_xs, scope='cifar', reuse=False)
        
        meta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='meta')
        mnist_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist')
        cifar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cifar')
        meta_vars.sort(key=lambda x: x.name)
        mnist_vars.sort(key=lambda x: x.name)
        cifar_vars.sort(key=lambda x: x.name)
        
        with tf.name_scope('losses'):
            self.assign_mnist = [x.assign(y) for x, y in zip(meta_vars, mnist_vars)]
            self.assign_cifar = [x.assign(y) for x, y in zip(meta_vars, cifar_vars)]
            
            mnist_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.mnist_input_ys, logits=self.mnist_adapt)
            cifar_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cifar_input_ys, logits=self.cifar_adapt)
            mnist_loss, cifar_loss = tf.reduce_mean(mnist_loss), tf.reduce_mean(cifar_loss)
            mnist_trainer = tf.train.AdamOptimizer(MNIST_LR)
            cifar_trainer = tf.train.AdamOptimizer(CIFAR_LR)
            meta_trainer = tf.train.AdamOptimizer(META_LR)
            
            mnist_grads_and_vars = mnist_trainer.compute_gradients(mnist_loss, var_list=mnist_vars)
            cifar_grads_and_vars = cifar_trainer.compute_gradients(cifar_loss, var_list=cifar_vars)
            meta_grads_and_vars = []
            for mgv, cgv, mv in zip(mnist_grads_and_vars, cifar_grads_and_vars, meta_vars):
                gradient_sum_for_meta = mgv[0] + cgv[0]
                meta_grads_and_vars.append((gradient_sum_for_meta, mv))
            self.mnist_optim = mnist_trainer.apply_gradients(mnist_grads_and_vars)
            self.cifar_optim = cifar_trainer.apply_gradients(cifar_grads_and_vars)
            self.meta_optim = meta_trainer.apply_gradients(meta_grads_and_vars)

            mnist_pred = tf.argmax(self.mnist_adapt, axis=-1, output_type=tf.int32)
            cifar_pred = tf.argmax(self.cifar_adapt, axis=-1, output_type=tf.int32)
            self.mnist_map = tf.reduce_mean(tf.cast(tf.equal(self.mnist_input_ys, mnist_pred), tf.float32))
            self.cifar_map = tf.reduce_mean(tf.cast(tf.equal(self.cifar_input_ys, cifar_pred), tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        #self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] Build meta model')
        
    def single_step_update(self):
        self.sess.run(self.assign_mnist)
        self.sess.run(self.assign_cifar)
        mnist_xs, mnist_ys = mnist_reader.next_batch(BATCH_SIZE)
        cifar_xs, cifar_ys = cifar_reader.next_batch(BATCH_SIZE)
        mnist_feed = {self.mnist_input_xs: mnist_xs, self.mnist_input_ys: mnist_ys}
        cifar_feed = {self.cifar_input_xs: cifar_xs, self.cifar_input_ys: cifar_ys}
        self.sess.run(self.mnist_optim, feed_dict=mnist_feed)
        self.sess.run(self.cifar_optim, feed_dict=cifar_feed)
        # use mnist_feed.update as the meta feed_dict
        mnist_feed.update(cifar_feed)
        self.sess.run(self.meta_optim, feed_dict=mnist_feed)


class AdvMetaModel(MetaModelBase):
    def __init__(self):
        super(AdvMetaModel, self).__init__()
        self.meta = build_network(self.input_xs, scope='meta', reuse=False)
        self.mnist_adapt = build_network(self.mnist_input_xs, scope='mnist', reuse=False)
        self.cifar_adapt = build_network(self.cifar_input_xs, scope='cifar', reuse=False)

        meta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='meta')
        mnist_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist')
        cifar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cifar')
        print(' [*] Number of vars: MNIST: {}, CIFAR: {}'.format(len(mnist_vars), len(cifar_vars)))

        mnist_pred = tf.argmax(self.mnist_adapt, axis=-1, output_type=tf.int32)
        cifar_pred = tf.argmax(self.cifar_adapt, axis=-1, output_type=tf.int32)
        self.mnist_map = tf.reduce_mean(tf.cast(tf.equal(self.mnist_input_ys, mnist_pred), tf.float32))
        self.cifar_map = tf.reduce_mean(tf.cast(tf.equal(self.cifar_input_ys, cifar_pred), tf.float32))

        # Losses and assign variables
        self.assign_mnist = [x.assign(y) for x, y in zip(meta_vars, mnist_vars)]
        self.assign_cifar = [x.assign(y) for x, y in zip(meta_vars, cifar_vars)]
        
        mnist_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.mnist_input_ys, logits=self.mnist_adapt)
        cifar_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cifar_input_ys, logits=self.cifar_adapt)
        mnist_loss, cifar_loss = tf.reduce_mean(mnist_loss), tf.reduce_mean(cifar_loss)

        self.mnist_adv_examples = fgsm_op(self.mnist_input_xs, mnist_loss, epsilon=EPSILON)
        self.cifar_adv_examples = fgsm_op(self.cifar_input_xs, cifar_loss, epsilon=EPSILON)
        adv_mnist_adapt = build_network(self.mnist_adv_examples, scope='mnist', reuse=True)
        adv_cifar_adapt = build_network(self.cifar_adv_examples, scope='cifar', reuse=True)
        with tf.name_scope('adversarial'):
            adv_mnist_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.mnist_input_ys, logits=adv_mnist_adapt)
            adv_cifar_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cifar_input_ys, logits=adv_cifar_adapt)
            adv_mnist_loss, adv_cifar_loss = tf.reduce_mean(mnist_loss), tf.reduce_mean(cifar_loss)

            adv_mnist_pred = tf.argmax(adv_mnist_adapt, axis=-1, output_type=tf.int32)
            adv_cifar_pred = tf.argmax(adv_cifar_adapt, axis=-1, output_type=tf.int32)
            self.adv_mnist_map = tf.reduce_mean(tf.cast(tf.equal(self.mnist_input_ys, adv_mnist_pred), tf.float32))
            self.adv_cifar_map = tf.reduce_mean(tf.cast(tf.equal(self.cifar_input_ys, adv_cifar_pred), tf.float32))

        with tf.name_scope('losses'):
            mnist_trainer = tf.train.AdamOptimizer(MNIST_LR)
            cifar_trainer = tf.train.AdamOptimizer(CIFAR_LR)
            meta_trainer = tf.train.AdamOptimizer(META_LR)
            # Use SGD optimizer for adv because no m and v in SGD
            adv_mnist_trainer = tf.train.GradientDescentOptimizer(MNIST_LR)
            adv_cifar_trainer = tf.train.GradientDescentOptimizer(CIFAR_LR)
            adv_meta_trainer = tf.train.GradientDescentOptimizer(META_LR)
            
            mnist_grads_and_vars = mnist_trainer.compute_gradients(mnist_loss, var_list=mnist_vars)
            cifar_grads_and_vars = cifar_trainer.compute_gradients(cifar_loss, var_list=cifar_vars)
            adv_mnist_grads_and_vars = mnist_trainer.compute_gradients(adv_mnist_loss, var_list=mnist_vars)
            adv_cifar_grads_and_vars = cifar_trainer.compute_gradients(adv_cifar_loss, var_list=cifar_vars)
            meta_grads_and_vars, adv_meta_grads_and_vars = [], []
            for mgv, cgv, mv in zip(mnist_grads_and_vars, cifar_grads_and_vars, meta_vars):
                gradient_sum_for_meta = mgv[0] + cgv[0]
                meta_grads_and_vars.append((gradient_sum_for_meta, mv))
            for mgv, cgv, mv in zip(adv_mnist_grads_and_vars, adv_cifar_grads_and_vars, meta_vars):
                gradient_sum_for_meta = mgv[0] + cgv[0]
                adv_meta_grads_and_vars.append((gradient_sum_for_meta, mv))

            self.mnist_optim = mnist_trainer.apply_gradients(mnist_grads_and_vars)
            self.cifar_optim = cifar_trainer.apply_gradients(cifar_grads_and_vars)
            self.meta_optim = meta_trainer.apply_gradients(meta_grads_and_vars)

            self.adv_mnist_optim = adv_mnist_trainer.apply_gradients(adv_mnist_grads_and_vars)
            self.adv_cifar_optim = adv_cifar_trainer.apply_gradients(adv_cifar_grads_and_vars)
            self.adv_meta_optim = adv_meta_trainer.apply_gradients(adv_meta_grads_and_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        #self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] Build meta model')

    def single_step_update(self):
        self.sess.run(self.assign_mnist)
        self.sess.run(self.assign_cifar)
        mnist_xs, mnist_ys = mnist_reader.next_batch(BATCH_SIZE)
        cifar_xs, cifar_ys = cifar_reader.next_batch(BATCH_SIZE)
        mnist_feed = {self.mnist_input_xs: mnist_xs, self.mnist_input_ys: mnist_ys}
        cifar_feed = {self.cifar_input_xs: cifar_xs, self.cifar_input_ys: cifar_ys}
        self.sess.run(self.mnist_optim, feed_dict=mnist_feed)
        self.sess.run(self.adv_mnist_map, feed_dict=mnist_feed)
        self.sess.run(self.cifar_optim, feed_dict=cifar_feed)
        self.sess.run(self.adv_cifar_optim, feed_dict=cifar_feed)
        # use mnist_feed.update as the meta feed_dict
        mnist_feed.update(cifar_feed)
        self.sess.run(self.meta_optim, feed_dict=mnist_feed)
        self.sess.run(self.adv_meta_optim, feed_dict=mnist_feed)


if __name__ == '__main__':
    if TEST_MODE:
        model = AdvMetaModel()
        saver = tf.train.Saver()
        saver.restore(model.sess, './ckpt/model')
        random_number = np.random.randint(0, 10000)
        for _ in range(random_number // 100):
            xs, ys = mnist_reader.test_batch(1)
        xss, yss = cifar_reader.test_batch(random_number, random_number + 1)
        mnist_feed = {model.mnist_input_xs: xs, model.mnist_input_ys: ys}
        cifar_feed = {model.cifar_input_xs: xss, model.cifar_input_ys: yss}
        mnist_adv_xs = model.sess.run(model.mnist_adv_examples, feed_dict=mnist_feed)
        cifar_adv_xs = model.sess.run(model.cifar_adv_examples, feed_dict=cifar_feed)
        mnist_feed[model.mnist_input_xs] = mnist_adv_xs
        cifar_feed[model.cifar_input_xs] = cifar_adv_xs
        mnist_map = model.sess.run(model.mnist_map, feed_dict=mnist_feed)
        cifar_map = model.sess.run(model.cifar_map, feed_dict=cifar_feed)

        print(' [*]', mnist_adv_xs.shape, mnist_adv_xs.min(), mnist_adv_xs.max())
        print(' [*]', cifar_adv_xs.shape, cifar_adv_xs.min(), cifar_adv_xs.max())

        plt.figure()
        plt.subplot(221)
        plt.imshow(xs[0])
        plt.subplot(222)
        plt.title('mAP: {}'.format(mnist_map))
        plt.imshow(mnist_adv_xs[0])
        plt.subplot(223)
        plt.imshow(xss[0])
        plt.subplot(224)
        plt.imshow(cifar_adv_xs[0])
        plt.title('mAP: {}'.format(cifar_map))
        plt.show()

        exit(0)

    else:
        with tf.variable_scope('meta-learning'):
            meta_medel = MetaModel()
        with tf.variable_scope('adv-meta-learning'):
            adv_meta_model = AdvMetaModel()

    meta_model.load_state(META_SAVE_DIR)
    adv_meta_model.load_state(ADV_SAVE_DIR)

    mnist_summaries, cifar_summaries = [], []
    x_steps = list(range(0, MAX_TRAIN_ITER, 100))
    for idx in range(MAX_TRAIN_ITER):
        meta_model.single_step_update()
        adv_meta_model.single_step_update()
        if idx % 100 == 0:
            #mnist_xs, mnist_ys = mnist_reader.test_batch(512)
            #cifar_xs, cifar_ys = cifar_reader.test_batch(0, 512)
            mnist_map = meta_model.evaluate_mnist()
            cifar_map = meta_model.evaluate_cifar()
            mnist_summaries.append(mnist_map)
            cifar_summaries.append(cifar_map)
            #mnist_map_train = meta_model.sess.run(meta_model.mnist_map, feed_dict={meta_model.input_xs: mnist_xs, meta_model.input_ys: mnist_ys})
            #cifar_map_train = meta_model.sess.run(meta_model.cifar_map, feed_dict={meta_model.input_xs: cifar_xs, meta_model.input_ys: cifar_ys})
            print('Iteration: {}, MNIST: {}, CIFAR-10: {}'.format(idx, mnist_map, cifar_map))
    meta_model.save_state(SAVE_DIR)

    plt.style.use('ggplot')
    #plt.figure()
    plt.plot(x_steps, mnist_summaries, linewidth=1.5)
    plt.plot(x_steps, cifar_summaries, linewidth=1.5)
    plt.legend(['MNIST', 'CIFAR'], loc='best')
    plt.xlabel('Training iterations')
    plt.ylabel('Precision on test set')
    plt.savefig('./images/meta_training.png')
    #plt.show()
