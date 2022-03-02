from scapy.all import *
load_layer("tls")

MYPC = "192.168.1.61"

def build(time,packets,srcs):
    t1 = packets[0].time + time
    section = set()
    d = {}
    nodes = set()
    for packet in packets:
        if packet.time < t1:
            try:
                if packet[IP].src in srcs and packet[IP].dst not in srcs:
                    if TCP in packet:
                        section.add((packet[IP].dst, packet[TCP].dport))
                    elif UDP in packet:
                        section.add((packet[IP].dst, packet[UDP].dport))
                elif packet[IP].src not in srcs and packet[IP].dst in srcs:
                    if TCP in packet:
                        section.add((packet[IP].src, packet[TCP].sport))
                    elif UDP in packet:
                        section.add((packet[IP].src, packet[UDP].sport))

                # if TCP in packet:
                #     if packet[IP].dst not in srcs: # destenation is not mypc
                #         section.add((packet[IP].dst,packet[TCP].dport))
                #     elif packet[IP].dst in srcs: # destenation is my pc
                #         section.add((packet[IP].src,packet[TCP].sport))
                # elif UDP in packet:
                #     if packet[IP].dst not in srcs:
                #         section.add((packet[IP].dst,packet[UDP].dport))
                #     elif packet[IP].dst in srcs:
                #         section.add((packet[IP].src,packet[UDP].sport))
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
            if packet[IP].src in srcs and packet[IP].dst not in srcs:
                if TCP in packet:
                    section.add((packet[IP].dst, packet[TCP].dport))
                elif UDP in packet:
                    section.add((packet[IP].dst, packet[UDP].dport))
            elif packet[IP].src not in srcs and packet[IP].dst in srcs:
                if TCP in packet:
                    section.add((packet[IP].src, packet[TCP].sport))
                elif UDP in packet:
                    section.add((packet[IP].src, packet[UDP].sport))
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
    