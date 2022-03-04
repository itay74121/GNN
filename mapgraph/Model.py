import tensorflow as tf
import numpy as np
from SniFiltering import *
from scapy.all import *
from glob import glob
from MapGraphBuild import build

import pandas as pd
from os import path
from statsmodels import robust
from scipy.stats import skew,kurtosis,scoreatpercentile
from random import choice,choices,randint
from sklearn.model_selection import train_test_split


class GCN(tf.keras.layers.Layer):
    def __init__(self,out,d) -> None:
        super().__init__()
        self.out = out
        self.d = d
    def build(self,input_shape):
        self.w = self.add_weight(shape=(self.d, self.out),
                               initializer='random_normal',
                               trainable=True)
    def call(self,inputs):
        return tf.nn.tanh(tf.matmul(inputs,self.w))

class MAPModel(tf.keras.Model):
    def __init__(self,out,N) -> None:
        super().__init__()
        self.N = N
        self.out = out
        self.GCN1 = GCN(1024,64)
        self.GCN2 = GCN(1024,1024)
        self.GCN3 = GCN(1024,1024)
        self.GCN4 = GCN(512,1024)
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
        s = inputs.shape[2]
        D = tf.slice(inputs[0][0],[0,0],[s,s])
        A = tf.slice(inputs[0][1],[0,0],[s,s])
        X = tf.slice(inputs[0][2],[0,0],[s,64])
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
        return t


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
    for i in range(l):
        m[i][i] = 1
    del l, d1, d, nodes
    return m

def getdegree(A):
    d = np.zeros(shape=A.shape)
    for i in range(len(A)):
        d[i][i] = np.sum(A[i])
    return d


def extract_features(node,packets):
    pdata = []
    t,u = 0,0
    for i in packets:
        if TCP in i:
            t+=1
        if UDP in i:
            u+=1
        try:
            if node in (i[IP].src,i[IP].dst):
                pdata.append((len(bytes(i)),i.time,i[IP].src,i[IP].dst))
        except:
            pass
    df = pd.DataFrame(pdata)
    features = [0]*64
    a = df[df[2] == node][0]
    b = df[df[2] != node][0]
    c = df[0]
    features[0] = len(df[df[2] == node])
    features[1] = max(a)
    features[2] = min(a)
    features[3] = np.mean(a)
    features[4] = robust.mad(a)
    features[5] = np.std(a)
    features[6] = np.var(a)
    features[7] = skew(a)
    features[8] = kurtosis(a)
    features[9] = scoreatpercentile(a,10)
    features[10] = scoreatpercentile(a, 20)
    features[11] = scoreatpercentile(a, 30)
    features[12] = scoreatpercentile(a, 40)
    features[13] = scoreatpercentile(a, 50)
    features[14] = scoreatpercentile(a, 60)
    features[15] = scoreatpercentile(a, 70)
    features[16] = scoreatpercentile(a, 80)
    features[17] = scoreatpercentile(a, 90)
    features[18] = len(df[df[2] != node])
    features[19] = max(b)
    features[20] = min(b)
    features[21] = np.mean(b)
    features[22] = robust.mad(b)
    features[23] = np.std(b)
    features[24] = np.var(b)
    features[25] = skew(b)
    features[26] = kurtosis(b)
    features[27] = scoreatpercentile(b,10)
    features[28] = scoreatpercentile(b, 20)
    features[29] = scoreatpercentile(b, 30)
    features[30] = scoreatpercentile(b, 40)
    features[31] = scoreatpercentile(b, 50)
    features[32] = scoreatpercentile(b, 60)
    features[33] = scoreatpercentile(b, 70)
    features[34] = scoreatpercentile(b, 80)
    features[35] = scoreatpercentile(b, 90)
    features[36] = len(df)
    features[37] = max(c)
    features[38] = min(c)
    features[39] = np.mean(c)
    features[40] = robust.mad(c)
    features[41] = np.std(c)
    features[42] = np.var(c)
    features[43] = skew(c)
    features[44] = kurtosis(c)
    features[45] = scoreatpercentile(c, 10)
    features[46] = scoreatpercentile(c, 20)
    features[47] = scoreatpercentile(c, 30)
    features[48] = scoreatpercentile(c, 40)
    features[49] = scoreatpercentile(c, 50)
    features[50] = scoreatpercentile(c, 60)
    features[51] = scoreatpercentile(c, 70)
    features[52] = scoreatpercentile(c, 80)
    features[53] = scoreatpercentile(c, 90)
    ################# getting the flows ##############
    flows= []
    # size time src dst
    treshhold = 3
    start = df[1][0]+treshhold
    flow = []
    for i in df.iterrows():
        if i[1][1] <= start:
            flow.append((i[1][0],i[1][1]))
        else:
            start = i[1][1]+treshhold
            flows.append(flow)
            flow = [(i[1][0],i[1][1])]
    flows.append(flow)
    features[54] = np.mean([sum([i[0] for i in flow]) for flow in flows])
    features[55] = np.mean([np.int64(np.ceil(flow[-1][1]) - np.floor(flow[0][1])) for flow in flows])
    features[56] = np.std([np.int64(np.ceil(flow[-1][1])-np.floor(flow[0][1])) for flow in flows])
    features[57] = len(flows)
    features[58] = np.mean([len(flow) for flow in flows])
    features[59] = 1 if t>=u else 0
    t = [int(i)/255 for i in node.split(".")]
    features[60] = t[0]
    features[61] = t[1]
    features[62] = t[2]
    features[63] = t[3]
    return features



def pcaptocsv(pcapfile,nodescsv,edgescsv,names,id):
    packets = rdpcap(pcapfile)
    fpackets = filterpackets(names,packets)
    d,nodes = build(3,fpackets)
    nodefeatures = []
    c = 1
    for node in nodes:
        nodefeatures.append([c,node]+extract_features(node[0],packets)+[id])
        c+=1
    outfeatures  = pd.DataFrame.from_dict(nodefeatures)
    edges = []
    c = 1
    for edge in d:
        edges.append([c,edge[0],edge[1],d[edge],id])
        c+=1

    outedges = pd.DataFrame.from_dict(edges)
    if path.exists(nodescsv):
        dfnodes = pd.read_csv(nodescsv)
        dfnodes.append(outfeatures)
    else:
        dfnodes = outfeatures
    if path.exists(edgescsv):
        dfedges = pd.read_csv(edgescsv)
        dfedges.append(outedges)
    else:
        dfedges = outedges
    with open(nodescsv,'w') as f:
        dfnodes.to_csv(f)
    with open(edgescsv,'w') as f:
        dfedges.to_csv(f)


def split(data,p,d):
    train = []
    for i in range(int(p*len(data))):
        c = choice(data)
        while len(d[c]) > 20:
            c = choice(data)
        data.remove(c)
        train.append(c)
    test = list(data)
    return train,test



def split_classes(x,y,classratio):
    y = [tuple(i) for i in y]
    train_x, train_y, test_x, test_y = [],[],[],[]
    n = x.shape[0]
    c = {}
    for i in classratio:
        c[i] = []
    for i in range(n):
        c[y[i]].append(x[i])
    for i in c:
        data = c[i] # test to be
        chosen= []
        for b in range(int(len(data)*classratio[i])):
            j=randint(0,len(data)-1)
            chosen.append(data[j])
            data.pop(j)
        train_x.extend(chosen)
        train_y.extend([list(i)]*len(chosen))
        test_x.extend(data)
        test_y.extend([list(i)]*len(data))
    return train_x,test_x,train_y,test_y



def main():
    # files = glob("./dataset/*.pcap")
    # d = {}
    # for file in files:
    #     pre = file.split(".pcap")[0]
    #     if pre not in d:
    #         d[pre] = [file]
    #     else:
    #         d[pre].append(file)
    # c = 1
    # for pre in d:
    #     print(c,pre,len(d[pre]))
    #     c+=1
    # print(np.mean([len(d[i]) for i in d]))
    # print(set([i[10] for i in d]))
    #
    # tojson = {}
    # with open("sninames.json",'r',encoding='utf-8') as f:
    #     sni = load(f)
    # c=  1
    # for pre in d:
    #     print(c)
    #     c+=1
    #     print("start",pre)
    #     for file in d[pre]:
    #         ps = rdpcap(file)
    #         snilist = getsni(ps)
    #         for i in sni['youtube']:
    #             for j in snilist:
    #                 f = False
    #                 if i in j[0]:
    #                     tojson[pre] = 'youtube'
    #                     f = True
    #                     break
    #                 if f:
    #                     break
    #         if f:
    #             continue
    #         for i in sni['twitter']:
    #             for j in snilist:
    #                 f = False
    #                 if i in j[0]:
    #                     tojson[pre] = 'twitter'
    #                     f = True
    #                     break
    #                 if f:
    #                     break
    #         del ps
    #
    # with open("filetowebsite.json",'w') as f:
    #     dump(tojson,f)


    #
    # keys = [i for i in d]
    # traindata,test = split(keys,0.25,d)
    # classes = list(set([i.split("\\")[1][0].lower() for i in traindata]))
    # data= []
    # ans = []
    # c= 1
    # for pre in d:
    #     print(c)
    #     c+=1
    #     print("started",pre)
    #     packets = []
    #     srcs = set([h.replace("-",'.') for h in [[j for j in i.split("_") if j.count("-") == 3][0] for i in d[pre]]])
    #     for file in d[pre]:
    #         print("read",file)
    #         ps = rdpcap(file)
    #         for t in ps:
    #             packets.append(t)
    #         del ps
    #     packets.sort(key=lambda x: x[0].time < x[1].time)
    #     edges,nodes = build(3,packets,srcs)
    #     A = getmatrix(list(nodes),edges)
    #     D = getdegree(A)
    #     X = []
    #     for n in nodes:
    #         X.append(extract_features(n[0],packets))
    #         print("done feature",n)
    #     X = np.array(X)
    #     t = tf.ragged.stack([D, A, X]).to_tensor().numpy()
    #     np.save(pre,t)
    #     # letter = pre.split("\\")[1][0].lower()
    #     # t = []
    #     # for i in classes:
    #     #     if i == letter:
    #     #         t.append(1)
    #     #     else:
    #     #         t.append(0)
    #     # data.append(tf.ragged.stack([D,A,X]).to_tensor())
    #     # ans.append(t)
    #
    #     del packets,edges,nodes,A,D,X
    #
    #
    # data = tf.ragged.stack(data).to_tensor()
    # ans = tf.convert_to_tensor(ans,dtype=tf.int64)
    #
    # data = np.load("data.npy")
    # ans = np.load("ans.npy")
    # for row in data:
    #     D = row[0]
    #     for i in range(D.shape[0]):
    #         t = D[i][i]
    #         if t>0:
    #             D[i][i] = 1/np.sqrt(t)
    #
    #
    #

    #
    # np.save("data",data.numpy())
    # np.save("ans",ans.numpy())


    files = glob("./dataset/*.npy")
    classes = list(set([i[10] for i in files]))
    classes.sort()
    y = []
    data = []
    for file in files:
        data.append(np.load(file))
        v = [0]*len(classes)
        v[classes.index(file[10])] = 1
        y.append(v)


    data = tf.ragged.constant(data).to_tensor().numpy()
    for row in data:
        D = row[0]
        for i in range(D.shape[0]):
            t = D[i][i]
            if t>0:
                D[i][i] = 1/t

    data_train, data_test, labels_train, labels_test = split_classes(data,y,{(1,0,0):0.75,(0,1,0):0.55,(0,0,1):0.143})   #train_test_split(data,y,stratify=y,test_size=0.1)
    print([sum([i[j] for i in labels_train]) / len(labels_train) for j in range(3)])
    data_train = tf.constant(data_train)
    labels_train = tf.constant(labels_train)

    data_test = tf.constant(data_test)
    labels_test = tf.constant(labels_test)

    t = [sum([i[j] for i in y])/len(y) for j in range(3)]
    d = {0:t[0],1:t[1],2:t[2]}

    # traindataset =  np.load("train_x.npy")
    # trainans = np.load("train_y.npy")
    # val_x = np.reshape(np.load("val_x.npy"),[1,3,28,64])
    # val_y = np.load("val_y.npy")
    #
    model = MAPModel(len(classes),5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.e-5), loss=tf.keras.losses.MeanSquaredError(),metrics = ['accuracy',"categorical_accuracy"])
    model.fit(x=data_train,y=labels_train,batch_size=1,epochs=8,class_weight=d)
    results = model.evaluate(x=data_test,y= labels_test, verbose=1)
    print(results)

if __name__ == "__main__":
    main()