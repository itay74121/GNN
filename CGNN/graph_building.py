from scapy.all import *
import networkx as nx 

def complete(s,l=1500):
    return tuple([i for i in s]+[0]*(l-len(s)))
def create_graph(filename):
    V = []
    file  = rdpcap(filename)
    for p in file:
        try:
            b = bytes(p)
            if len(b)<=1500:
                V.append((float(p.time),complete(b)))
            
        except:
            pass
    return V
    
vectors = create_graph("./data/graph2.pcap")

eps = 0.001
chains = []
chain = []
for i in range(len(vectors)-1):
    a,b = vectors[i],vectors[i+1]
    if b[0]-a[0] < eps:
        chain.append(a)
        chain.append(b)
    else:
        chain.append(a)
        chains.append(chain)
        chain = []
        
        
with open("graph.txt",'w') as f:
  for chain in chains:
      for i in range(len(chain)-1):
          if len(chain) > 1:
              f.write(str(chain[i][1]).replace(" ","")+" "+str(chain[i+1][1]).replace(" ","")+" 1\n")
          else:
              f.write(str(chain[i][1]).replace(" ","")+str(chain[i][1]).replace(" ","")+" 1\n")


    
