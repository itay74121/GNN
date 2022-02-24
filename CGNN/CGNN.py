from scapy.all import *
import numpy as np
from scapy.layers.inet import TCP, Ether, IP, UDP
from scapy.layers.dns import DNS
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
from sklearn.model_selection import train_test_split


class SgcLayer(tfk.layers.Layer):
    def __init__(self, outputsNumber):
        super(SgcLayer, self).__init__()
        self.outputs = outputsNumber
        pass

    def build(self, input_shape):
        self.teta = self.add_weight("teta",
                                    shape=[input_shape[-1], self.outputs],
                                    trainable=True,
                                    initializer="random_normal")

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.teta))


class cgnn_model(tf.keras.Model):
    def __init__(self):
        super(cgnn_model, self).__init__()
        self.sgc1 = SgcLayer(516)
        self.sgc2 = SgcLayer(256)
        self.dense = tf.keras.layers.Dense(6)

    def call(self, inputs):
        x = self.sgc1(inputs)
        x = self.sgc2(x)
        x = tf.keras.layers.AveragePooling1D(strides=1, pool_size=(inputs.shape[1]))(x)
        x = self.dense(x)
        x = tf.nn.softmax(x)
        return x


def get_diagonal_degree(A):
    D = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        D[i][i] = np.sum(A[i])
    return D


def sum_matrices(A, B):
    result = np.array([[0 for x in range(len(A))] for y in range(len(A))])
    for i in range(len(A)):
        # iterate through columns
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
    return np.array(result)


def normalize_D_t(D_t):
    for i in range(len(D_t)):
        D_t[i][i] = 1 / np.sqrt(D_t[i][i])
    return D_t



def getSX(session):
    X = np.array(session)
    A = calculate_A(session)
    I = np.identity(len(A[0]))
    A_t = sum_matrices(A, I)
    D_t = np.array(get_diagonal_degree(A_t))
    D_t = normalize_D_t(D_t)
    S_temp = np.dot(D_t, A_t)
    S = np.dot(S_temp, D_t)
    SX = np.dot(S, X)
    return np.asarray(SX)


def complete(s, l=1500):
    p = [i for i in s] + [0] * (l - len(s))
    return p[:1500]


def calculate_A(session):
    X = session
    A = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X)):
            if i == j + 1 or i == j - 1:
                temp.append(1)
            else:
                temp.append(0)
        A.append(temp)
    return np.array(A)


def createGraphFromSession(pcapName):
    file = rdpcap("data/w_hi_chrome/" + pcapName)
    l = []
    for p in file:
        packet_proc = preprocessing(p)
        if packet_proc:
            if complete(packet_proc):
                l.append(complete(packet_proc))
    return getSX(l)


def preprocessing(packet):
    if packet.haslayer(IP):
        packet[IP].src = 0
        packet[IP].dst = 0
        if packet.haslayer(TCP):
            FIN = 0x01
            SYN = 0x02
            ACK = 0x10
            F = packet['TCP'].flags  # this should give you an integer
            if F & FIN or F & SYN or F & ACK or packet.haslayer(DNS):
                if len(packet) <= 66:
                    return None

            w_eth_header = bytes(packet)[14:]
            return w_eth_header
        elif packet.haslayer(UDP):
            w_eth_header = bytes(packet)[14:]
            zero_bytes = bytearray(12)
            new_packet = bytes(w_eth_header[:8]) + zero_bytes + bytes(w_eth_header[8:])
            return new_packet


def getTrainTestGraphs():
    df = pd.read_csv("data/w_hi_chrome/id.csv")
    train_name, test_name, train_label, test_label = train_test_split(df["fname"],
                                                                      df["label"],
                                                                      test_size=0.20,
                                                                      random_state=42)
    GraphsForTrain = []
    GraphsForTest = []

    for i in train_name:
        g = createGraphFromSession(i)

        GraphsForTrain.append(np.ndarray.tolist(g))

    LableForTrain = []
    for i in train_label:
        LableForTrain.append(int(i))

    for name in df["fname"]:
        if name not in train_name:
            g = createGraphFromSession(name)
            GraphsForTest.append(g)

    return GraphsForTrain, list(LableForTrain), GraphsForTest, list(test_label)  # the graphs is SX


GraphsForTrain, LabelsForTrain, GraphsForTest, LabelsForTest = getTrainTestGraphs()
m = cgnn_model()
m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
          loss='categorical_crossentropy',metrics=['acc'])

max_n = max([len(i) for i in GraphsForTrain])
for g in GraphsForTrain:
    zeros = [0] * 1500
    for i in range(max_n - len(g)):
        g.append(zeros)

graph_to_train = tf.convert_to_tensor(GraphsForTrain, tf.float32)
graph_to_train = tf.cast(graph_to_train, tf.float32)

dict_label = {}
counter = 0
for item in set(LabelsForTrain):
    dict_label[item] = counter
    counter += 1


list_of_lables = []
for item in LabelsForTrain:
    list_temp = [0] * 6
    list_temp[dict_label[item]] = 1
    list_of_lables.append(list_temp)

label_graph_to_train = tf.convert_to_tensor(list_of_lables)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
m.fit(graph_to_train, np.reshape(label_graph_to_train, (2604, 1, 6)), epochs=400, callbacks=[callback], batch_size=32)

list_of_lables_test = []
for item in LabelsForTest:
    list_temp = [0] * 6
    list_temp[dict_label[item]] = 1

    list_of_lables_test.append(list_temp)
for g in GraphsForTest:
    zeros = [0] * 1500
    for i in range(max_n - len(g)):
        np.append(g, zeros)
graph_to_test = tf.convert_to_tensor(GraphsForTrain, tf.float32)
graph_to_test = tf.cast(graph_to_test, tf.float32)

m.evaluate(graph_to_test, np.reshape(list_of_lables_test, (651, 1, 6)))
