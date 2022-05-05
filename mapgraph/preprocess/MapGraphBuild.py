from BuildGraph import *
from SniFiltering import *
from scapy.all import *



def main():
    filename = "instagram.pcap"
    name = "inst"
    eps = 1 # one second
    packets = rdpcap(filename)
    packets = filterpackets(name,packets)
    d,nodes = build(eps,packets)
    with open("instagram.txt",'w') as f:
        for i in d:
            f.write(f"{str(i[0]).replace(' ','')} {str(i[1]).replace(' ','')} {d[i]}\n")
    
    
if __name__ == "__main__":
    main()