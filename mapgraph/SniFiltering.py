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
    if type(name) is list:
        l= []
        for n in name:
            for sni in snilist:
                if n in sni[0]:
                    l.append(sni)
        l = list(set(l))

    else:
        l = []
        for sni in snilist:
            if name in sni[0]:
                l.append(sni)
    packs= []
    for ip in l:
        print(ip,len(packs))
        for p in packets:
            try:
                iph = p[IP]
                if ip[1] in(iph.dst):
                    packs.append(p)
            except:
                pass
    packs.sort(key = lambda x:x[0].time<x[1].time)
    return packs