import tensorflow as tf
import numpy as np
import grequests as greq
import threading
from time import sleep
from datetime import datetime

class NetworkDataset(object):
    def __init__(self, batch_size, capacity, shape, url_list):
        self.batch_size = batch_size
        self.mempool = np.zeros([capacity] + shape, dtype=np.float32)
        self.end_ptr = 0
        self.size = 0
        self.capacity = capacity
        self.url_list = url_list
        self.mutex = threading.Lock()
        
        self.thread = threading.Thread(target=self.run_thread)
        self.thread.start()
        self.batch_data = tf.py_func(self.get_data, [], tf.float32)

    def set_data(self):
        req_pkgs = [greq.get(url) for url in self.url_list]
        resp_pkgs = greq.map(req_pkgs)
        resp_data = [resp.json() for resp in resp_pkgs if resp and resp.status_code == 200]
        array_data = [np.array(json['data']).reshape([-1] + shape) for json in resp_data]

        for array in array_data:
            n_sample = array.shape[0]
            with self.mutex:
                self.mempool[self.end_ptr: self.end_ptr + n_sample] = array
            self.end_ptr += n_sample
            if self.end_ptr >= self.capacity:
                self.end_ptr = 0
            self.size += n_sample

    def get_data(self):
        while self.size < self.capacity:
            print(' [*] Mempool not filled yet...')
            sleep(3)
        rand_indices = np.random.randint(0, self.capacity, [self.batch_size])
        #print('[DEBUG]', rand_indices)
        with self.mutex:
            data_array = self.mempool[rand_indices].copy()
        return data_array

    def run_thread(self):
        print(' [*] set data started in {} ...'.format(datetime.now()))
        while True:
            self.set_data()



if '__main__' == __name__:
    network_dataset = NetworkDataset(2, 128, [4], '')
    
    with tf.Session() as sess:
        batch_data = sess.run(network_dataset.batch_data)
        print(batch_data.shape, batch_data.dtype)

        batch_data = sess.run(network_dataset.batch_data)
        print(batch_data.shape, batch_data.dtype)
