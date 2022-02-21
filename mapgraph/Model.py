import tensorflow as tf
import numpy as np
from SniFiltering import *
from scapy.all import *
from glob import glob
from MapGraphBuild import build

class GCN(tf.keras.layers.Layer):
    def __init__(self,out) -> None:
        super().__init__()
        self.out = out
    def build(self,input_shape):
        self.w = self.add_weight(shape=(input_shape[-1][1], self.out),
                               initializer='random_normal',
                               trainable=True)
    def call(self,inputs):
        # D^-1,A,Z
        DAZ = tf.matmul(tf.matmul(inputs[0],inputs[1]),inputs[2])
        return inputs[0],inputs[1],tf.nn.tanh(tf.matmul(DAZ,self.w))

class MAPModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.GCN1 = GCN(1024)
        self.GCN2 = GCN(1024)
        self.GCN3 = GCN(1024)
        self.GCN4 = GCN(512)
        self.sortpooling = tf.keras.layers.sort
        

def getmatrix(nodes,d):
    l = len(nodes)
    d1 = {}
    for i in range(l):
        d1[nodes[i]] = i
    m = np.zeros((l, l))
    for edge in d:
        # edge =  (ni,nj)
        m[d1[edge[0]]][d1[edge[1]]] = d[edge]
        m[d1[edge[1]]][d1[edge[0]]] = d[edge]
    del l, d1, d, nodes
    return m

def main():
    # gcn = GCN(5)
    # X = np.random.rand(20,5)
    # D = np.random.rand(20,20)
    # A = np.random.rand(20,20)
    # i = gcn((D,A,X))
    #print(i)
    files = "./dataset/windows_chrome_twitter.pcap"
    packets = rdpcap(files)
    for i in getsni(packets):
        print(i)
    p = filterpackets(["twimg","twitter"],packets)
    print(p)
    # end = filterpackets(["amazon","cloudfront"],packets)
    # d,nodes = build(1,end)
    # with open("amazon.txt",'w') as f:
    #     for i in d:
    #         f.write(f"{str(i[0]).replace(' ', '')} {str(i[1]).replace(' ', '')} {d[i]}\n")


if __name__ == "__main__":
    main()