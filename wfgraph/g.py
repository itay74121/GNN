from scapy.all import *
filename = "pcapfile.pcap"
file  = rdpcap(filename)
type(file)

# file[0].psrc
# file[0].pdst
file[0].type 
# file[0].dst
# file[0].src
# file[0].ptype
file[0].show()
pac = file[0]
print(pac)
# print(pac.proto)

j = 0 
e = True
proto = PacketList()
scapy.plist.PacketList
if e == True:
    for i in range(len(file)):
        # if
        # print(i)
        # file[0].show()

        if(file[i].type == 2048 and file[i].proto == 6 and e):
            proto.append(file[i])
            

            j = i 
            #e = False
            # file[i].show

file[j].show()

# file[j].psrc
# file[0].pdst
# file[j].type
# file[j].proto
# file[j].dport
# file[0].dst
# file[0].src
# file[0].ptype
type(proto)
# proto[0].show()
print(len(proto))
# file[j].show()
# print(proto[0])

# print(proto[0][IP].dst)
proto[0].time


ing = 0
f = set()
ips = set()
for i in proto: 
    packetI = i
    sp = packetI.sport
    dp = packetI.dport
    prtcl = packetI.proto
    sIp = packetI[IP].src
    dIp = packetI[IP].dst
    tupleI = (sIp,dIp,sp,dp,prtcl)
    if ing == 0:
        print(tupleI)
        ing = ing + 1
    f.add(tupleI)
    # flows.append(flow)
    # flow = []

print(len(f))
# f
# print(f)
# f






flow = PacketList()
flows = {}

for i in f:
    for j in proto:
        fiveT = i
        sIp = fiveT[0]
        dIp = fiveT[1]
        sp = fiveT[2]
        dp = fiveT[3]
        prtcl = fiveT[4]
        if sIp == j[IP].src and dIp == j[IP].dst and sp == j.sport and dp == j.dport and prtcl == j.proto:
            flow.append(j)
    # flows.append(flow)
    flows[fiveT] = flow
    flow = PacketList()

print(len(flows))



keys = []
for i in flows.keys():
    keys.append(i)

threshold = 3
dictEdge ={}
for i in range(len(keys)-1):
    setlistI = []
    for j in flows.get(keys[i]):
        setlistI.append(j) 
    for j in range(i+1,len(keys)):
        setlistJ = []
        for t in flows.get(keys[j]):
            setlistJ.append(t)
        time1 = setlistI[0].time
        time2 = setlistJ[0].time
        src1 = setlistI[0][IP].src
        src2 = setlistJ[0][IP].src
        if abs(time1 - time2) < threshold:
            if src1 == src2:
                dictEdge[keys[i]] = keys[j]
                # dictEdge[keys[j]] = keys[i]

print(len(dictEdge))
# print(dictEdge)

import networkx as nx
import matplotlib.pyplot as plt

g = nx.DiGraph()
g.add_nodes_from(keys)
# print(keys)

for i in dictEdge.keys():
    g.add_edge(i ,dictEdge.get(i))
    g.add_edge(dictEdge.get(i) ,i)

nx.draw(g,with_labels=False)
plt.draw()
plt.show()






        
