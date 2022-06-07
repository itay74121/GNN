import numpy
from scapy.all import *
import numpy as np
from scapy.layers.inet import TCP, Ether, IP, UDP
from scapy.layers.dns import DNS
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
from tensorflow.keras import activations
from sklearn.model_selection import train_test_split
import keras.backend as K
import glob
import sys
numpy.set_printoptions(threshold=sys.maxsize)
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class cgnn_model(tfk.Model):
    def __init__(self, n_classes=79):
        super(cgnn_model, self).__init__()
        self.sgc1 = SgcLayer(516, 1, 'relu')
        self.sgc2 = SgcLayer(256, 1, 'relu')
        self.dense = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x = self.sgc1(inputs)
        x = self.sgc2(x)
        x = tf.keras.layers.AveragePooling1D(strides=1, pool_size=(inputs.shape[1]))(x)
        x = self.dense(x)
        x = tf.nn.softmax(x)
        return x


class SgcLayer(tfk.layers.Layer):
    def __init__(self, outputsNumber, neighbourhood_distance=1, activation=None):
        super(SgcLayer, self).__init__()
        self.outputs = outputsNumber
        self.S = None # local normalized adj matrix
        self.neighbourhood_distance = neighbourhood_distance
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.S = tf.Variable(
            initial_value=np.linalg.matrix_power(
                getS(input_shape[1]), 
                self.neighbourhood_distance
            ), 
            trainable=False,
            name='Normalized Adj. Matrix'
        )
        self.teta = self.add_weight(
            "teta",
            shape=[input_shape[2], self.outputs],
            trainable=True,
            initializer="random_normal"
        )

    def call(self, inputs):
        outputs = tf.matmul(self.S, float(inputs))
        outputs = tf.matmul(outputs, self.teta)
        if self.activation is not None:
            outputs = self.activation(outputs)
            
        return outputs

def getS(n):
    if n == 1:
        return np.array(0)
    if n == 2:
        return np.array([(0, 1),(1, 0)])
    S = np.zeros((n,n), dtype='float32')
    S[0,0], S[n - 1,n - 1] = 0.50000, 0.50000 # first and last vertices with self loop
    S[0,1], S[1,0], S[n - 2,n - 1], S[n - 1,n - 2] = 0.40825, 0.40825, 0.40825, 0.40825 # first and last vertices with other edges
    S[1,1], S[1,2], S[n - 2,n - 2], S[n - 2,n - 3] = 0.33333, 0.33333, 0.33333, 0.33333 # rest connected to first and last
    for i in range(2, len(S) - 2): # rest of vertices
        for j in range(i - 1, i + 2):
            S[i,j] = 0.33333
    return S


def complete(s, l=100):
    p = [i for i in s] + [0] * (l - len(s))
    return p[:100]


def createGraphFromSession(pcapName): # return X features matrix
    try:
        file = rdpcap(pcapName, count=300)
    except:
        return None
    l = []
    for p in file:
        if len(l) > 99:
            break
        packet_proc = preprocessing(p)
        if packet_proc:
            if complete(packet_proc):
                l.append(complete(packet_proc))
        # if len(l) < 4:
        #   return None
    del file, packet_proc
    if len(l) == 0:
        return None
    return l


def preprocessing(packet):
    if packet.haslayer(IP):
        packet[IP].src = 0
        packet[IP].dst = 0
        if packet.haslayer(TCP):
            FIN = 0x01
            SYN = 0x02
            ACK = 0x10
            F = packet['TCP'].flags  # this should give you an integer
            if F & FIN or F & SYN or F & ACK:
                if len(packet[TCP].payload) == 0:
                    return None

            w_eth_header = bytes(packet)[14:]
            return w_eth_header
        elif packet.haslayer(UDP):
            if packet.haslayer(DNS):
                if len(packet[DNS].payload) == 0:
                    return None
            w_eth_header = bytes(packet)[14:]
            zero_bytes = bytearray(12)
            new_packet = bytes(w_eth_header[:8]) + zero_bytes + bytes(w_eth_header[8:])
            return new_packet


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return recall


def TP(y_true, y_pred):
    tp = tf.math.count_nonzero(y_pred * y_true)
    return tp


def TN(y_true, y_pred):
    tn = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
    return tn


def FP(y_true, y_pred):
    fp = tf.math.count_nonzero(y_pred * (y_true - 1))
    return fp


def FN(y_true, y_pred):
    fn = tf.math.count_nonzero((y_pred - 1) * y_true)
    return fn


def main():
 

    # ====================================================================================
    files_names = glob.glob("./Mapp Graph/*/*/*.npy")
    labels = [int(i.split("pcap_")[1].split(".")[0]) for i in files_names]
    print("debug")
    train_name, test_name, train_label, test_label = train_test_split(files_names, labels, test_size=0.2, random_state=42, stratify = labels)

    classes = list(set(labels))
    classes.sort()
    train_l = []
    for i in train_label:
        v = [0] * len(classes)
        v[classes.index(i)] = 1
        train_l.append(v)

    test_l = []
    for i in test_label:
        v = [0] * len(classes)
        v[classes.index(i)] = 1
        v[classes.index(i)] = 1
        test_l.append(v)
    train_l = tf.expand_dims(tf.constant(train_l), axis=1)
    test_l = tf.expand_dims(tf.constant(test_l), axis=1)
    print("before load")
    # load the files
    train = []
    test = []
    for i in train_name:
        print("loading file: ", i)
        train.append(np.load(i))
    for i in test_name:
        print("loading label: ", i)
        test.append(np.load(i))
    print("finished loading")
    train_ = []
    for i in train:
        tf_m = tf.constant(i)
        if tf_m.shape[0] < 100:
            padding = tf.constant([[0, 100-tf_m.shape[0]], [0, 0]])
            tf_m = tf.pad(tf_m, padding)
        train_.append(tf_m)
    train = tf.stack(train_)
    print("finished train padding")

    test_ = []
    for i in test:
        tf_m = tf.constant(i)
        if tf_m.shape[0] < 100:
            padding = tf.constant([[0, 100-tf_m.shape[0]], [0, 0]])
            tf_m = tf.pad(tf_m, padding)
        test_.append(tf_m)
    test = tf.stack(test_)
    print("finished test padding")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
    m = cgnn_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['acc', f1, precision, recall, TP, TN, FP, FN])
    m.fit(x=train, y=train_l, batch_size=32, epochs= 400, shuffle = True, callbacks=[callback])
    # test:
    m.evaluate(test, test_l)

    a = np.array(m.predict(test))
    idx = np.argmax(a, axis=-1)
    idx = idx.flatten()
    b = np.array(test_l)
    _idx = np.argmax(b, axis=-1)
    _idx = _idx.flatten()

    result = tf.math.confusion_matrix(labels=_idx, predictions=idx,
                                      num_classes=79)  # use one hot to convert to 1D to do confMatrix
    print(result)

if __name__ == "__main__":
    main()