from BuildGraph import build
from scapy.all import * 
import pandas as pd
import numpy as np
from statsmodels import robust
from scipy.stats import skew,scoreatpercentile,kurtosis
features = ['IP_port', 'complete_max', 'complete_min', 'complete_mean', 'complete_mad', 'complete_std', 'complete_var', 'complete_skew', 'complete_kurt', 'complete_pkt_num', 'complete_10per', 'complete_20per', 'complete_30per', 'complete_40per', 'complete_50per', 'complete_60per', 'complete_70per', 'complete_80per', 'complete_90per', 'out_max', 'out_min', 'out_mean', 'out_mad', 'out_std', 'out_var', 'out_skew', 'out_kurt', 'out_pkt_num', 'out_10per', 'out_20per', 'out_30per', 'out_40per', 'out_50per', 'out_60per', 'out_70per', 'out_80per', 'out_90per', 'in_max', 'in_min', 'in_mean', 'in_mad', 'in_std', 'in_var', 'in_skew', 'in_kurt', 'in_pkt_num', 'in_10per', 'in_20per', 'in_30per', 'in_40per', 'in_50per', 'in_60per', 'in_70per', 'in_80per', 'in_90per', 'protocol', 'flows_num', 'flow_length_mean', 'flow_pkt_num_mean', 'flow_duration_mean', 'ip1', 'ip2', 'ip3', 'ip4', 'graph_id']

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
    features[1] = max(a) if len(a)>0 else 0
    features[2] = min(a) if len(a)>0 else 0
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
    features[19] = max(b) if len(b)>0 else 0
    features[20] = min(b) if len(b)>0 else 0
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
    features[37] = max(c) if len(c)>0 else 0
    features[38] = min(c) if len(c)>0 else 0
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
    return features[1:]

def findsrc(packets):
    d = {}
    for i in packets:
        d.setdefault(i[IP].src,0)
        d[i[IP].src] += 1
    return max(d,key=d.get)

def pcap2csvs(pcap:str,timeinterval:int,srcs:tuple):
    '''
        @pcap - file name of the pcap file.
        @srsc - A tuple of all the srcs IP's in the file
        RETURN 
        The function returns a tuple of two data frames,
        the first data frame contains the node features, and the
        second dataframe contains the edges.
    '''
    packets = [i for i in rdpcap(pcap)] # extract packets into list 
    srcs = [findsrc(packets)]
    packets.sort(key=lambda x: x.time) # sort according to time.   NOT TAKING CHANCES
    edges,nodes = build(timeinterval,packets,srcs)
    nodes = list(nodes)
    temp = []#pd.DataFrame(columns=['source','target','weight','id'])
    c = 0
    for edge in edges:
        temp.append([edge[0],edge[1],edges[edge],c])
    edges = pd.DataFrame(temp,columns=['source','target','weight','id'])
    X = []
    for n in nodes:
        temp = [str(n)] + extract_features(n[0],packets)
        temp.append(c)
        X.append(temp)
        print("done feature",n)
    nodes = pd.DataFrame(X,columns=features)
    
    return nodes,edges


def pcaps2csvs(pcapslist,interval,srcslist):
    l = []
    for i in range(len(pcapslist)):
        print(pcapslist[i])
        t = pcap2csvs(pcapslist[i],interval,srcslist[i])
        t[0].to_csv(pcapslist[i]+".features.csv")
        t[1].to_csv(pcapslist[i]+".edges.csv")
        l.append(t)
    return l



def main():
    pass
if __name__ == "__main__":
    main()


