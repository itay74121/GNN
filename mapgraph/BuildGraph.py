from scapy.all import *
load_layer("tls")

MYPC = "192.168.1.61"

def build(time,packets):
    t1 = packets[0].time + time
    section = set()
    d = {}
    nodes = set()
    for packet in packets:
        if packet.time < t1:
            try:
                if TCP in packet:
                    if MYPC != packet[IP].dst:
                        section.add((packet[IP].dst,packet[TCP].dport))
                    elif MYPC == packet[IP].dst:
                        section.add((packet[IP].src,packet[TCP].sport))
                elif UDP in packet:
                    if MYPC != packet[IP].dst:
                        section.add((packet[IP].dst,packet[UDP].dport))
                    elif MYPC == packet[IP].dst:
                        section.add((packet[IP].src,packet[UDP].sport))
            except:
                pass
        else:
            l = list(section)
            for i in range(len(l)):
                for j in range(i+1,len(l)):
                    ni = l[i]
                    nj = l[j]
                    if (ni,nj) in d:
                        d[(ni,nj)]+=1
                    elif (nj,ni) in d:
                        d[(nj,ni)]+=1
                    else:
                        d[(ni,nj)] = 1     
            nodes = set.union(nodes,section)
            section = set()
            if TCP in packet:
                if MYPC != packet[IP].dst:
                    section.add((packet[IP].dst,packet[TCP].dport))
                elif MYPC == packet[IP].dst:
                    section.add((packet[IP].src,packet[TCP].sport))
            elif UDP in packet:
                if MYPC != packet[IP].dst:
                    section.add((packet[IP].dst,packet[UDP].dport))
                elif MYPC == packet[IP].dst:
                    section.add((packet[IP].src,packet[UDP].sport))
            t1 = packet.time + time
    l = list(section)
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            ni = l[i]
            nj = l[j]
            if (ni,nj) in d:
                d[(ni,nj)]+=1
            elif (nj,ni) in d:
                d[(nj,ni)]+=1
            else:
                d[(ni,nj)] = 1     
    nodes = set.union(nodes,section)
    return d,nodes

# d,nodes = devider(1,ps)
# d,nodes

def main():
    pass
if __name__ == "__main__":
    main()
    