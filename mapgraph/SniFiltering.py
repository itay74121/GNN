from scapy.all import *
load_layer("tls")

def getsni(packets):
    snilist = []
    for p in packets:
        if TLS in p:
            if p[TLS].type == 22:
                try:
                    extensions = p[TLS].msg[0].ext
                    for ext in extensions:
                        if ext.type == 0: # found sni field
                            snilist.append((ext.servernames[0].servername.decode(), p[IP].dst))
                except:
                    pass
    return snilist

def filterpackets(name,packets):
    snilist = set(getsni(packets))
    l = list(filter(lambda x: name in x[0],snilist))
    print(l)
    packs= []
    for ip in l:
        print(ip,len(packs))
        for p in packets:
            try:
                iph = p[IP]
                if ip[1] in(iph.src,iph.dst):
                    packs.append(p)
            except:
                pass
    packs.sort(key = lambda x:x[0].time<x[1].time)
    return packs