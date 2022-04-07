import dgl
import tensorflow as tf
from dgl.nn.tensorflow import GraphConv
import glob
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split

class GCN(tf.keras.layers.Layer):
    def __init__(self,out) -> None:
        super().__init__()
        self.out = out
    def build(self,input_shape):
        self.w = self.add_weight(name='weight',shape=(input_shape[-1], self.out),
                               initializer='random_normal',
                               trainable=True)
    def call(self,inputs):
        return tf.nn.tanh(tf.matmul(inputs,self.w))



class SortPooling(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super(SortPooling, self).__init__()
    def call(self,inputs):
        pass

class MAPModel(tf.keras.Model):
    def __init__(self,out) -> None:
        super().__init__()
        self.out = out
        self.GCN1 = GCN(1024)
        self.GCN2 = GCN(1024)
        self.GCN3 = GCN(1024)
        self.GCN4 = GCN(512)
        self.conv256 = tf.keras.layers.Conv1D(filters=256,kernel_size=1)
        self.maxpool = tf.keras.layers.MaxPool1D(pool_size=2)
        self.conv512 = tf.keras.layers.Conv1D(filters=512, kernel_size=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1024 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.drop25 = tf.keras.layers.Dropout(rate=0.25)
        self.dense8 = tf.keras.layers.Dense(units=self.out, activation="softmax")
    def call(self,inputs):
        """
        inputs are D and A and Z
        """
        # ans = []
        # for i in range(inputs.shape[0]):
        work = inputs[0]
        s = inputs.shape[2]
        D = tf.slice(work[0],[0,0],[s,s])
        A = tf.slice(work[1],[0,0],[s,s])
        X = tf.slice(work[2],[0,0],[s,64])
        t = tf.matmul(tf.matmul(D,A),X)
        t = self.GCN1(t)
        t = tf.matmul(tf.matmul(D, A), t)
        t = self.GCN2(t)
        t = tf.matmul(tf.matmul(D, A), t)
        t = self.GCN3(t)
        t = tf.matmul(tf.matmul(D, A), t)
        t = self.GCN4(t)
        t = tf.stack([t])
        t = self.conv256(t)
        t = self.maxpool(t)
        t = self.conv512(t)
        t = self.flatten(t)
        t = self.dense1024(t)
        t = self.drop25(t)
        t = self.dense8(t)
        #     ans.append(t[0])
        # ans = tf.stack(ans)
        return t



def main():
    # load data
    files = glob.glob("./FS raw/*/*.npy")
    classes = list(set([int(i.split("_")[-1].split('.npy')[0]) for i in files]))
    classes.sort()
    data = []
    y= []
    for file in files:
        label = int(file.split("_")[-1].split('.npy')[0])
        v = [0] * len(classes)
        v[classes.index(label)] = 1
        y.append(v)
        data.append(np.load(file))
    data_train, data_test, labels_train, labels_test = train_test_split(data,y,train_size=0.8)
    # graphs =[]
    # for graph in data:
    #     srcs = []
    #     dsts = []
    #     A = graph[1]
    #     for i in range(A.shape[0]):
    #         for j in range(A.shape[0]):
    #             if A[i][j] > 0:
    #                 srcs.append(i)
    #                 dsts.append(j)
    #     graphs.append(dgl.graph((srcs,dsts),num_nodes=A.shape[0]))

    # t = GraphConv(64,1024,weight=True).in_out_tensors()
    # print(t(graphs[0],tf.ones((4,64))).shape)
    m = MAPModel(len(classes))
    m.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

    m.fit(x=tf.ragged.constant(data_train).to_tensor(),y=tf.stack(labels_train),epochs=20,batch_size=1)

    print()
if __name__ == "__main__":
    main()

