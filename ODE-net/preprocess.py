from struct import pack
from scapy.all import *
from glob import glob
import pandas as pd
import numpy as np
import os
'''
preprocess for the BOA data set.
'''

path_to_dataset = 'D:\Infomedia_data_backups\Raw pcaps\\filtered_raw_dataset_temu2016_first_10_min'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    # retrive all folders
    folders = [i for i in glob(path_to_dataset+'\\*') if os.path.isdir(i)]
    print(folders)
    # start looping
    
    DATA = []
    LABELS = []
    for i,folder in enumerate(folders):
        print('start working on',i,folder)
        pcapfiles = glob(folder+'\\*.pcap')
        try:
            iddf = pd.read_csv(folder+'\\id.csv') # id dataframe for short
        except Exception as e: # failed reading id 
            print('    skiping',folder)
            continue
        for pcap in pcapfiles:
            print('    working on',pcap)
            flowsdict = {}
            packets = rdpcap(pcap)
            packets = list(packets)
            packets.sort(key=lambda x:x.time)
            time_bool = any([i<0 for i in np.diff([i.time for i in packets])])
            if time_bool:
                continue
            for packet in packets:
                t = (packet[IP].src,packet[IP].sport,packet[IP].dst,packet[IP].dport,packet[IP].proto)
                # check if key exist if not create one for the flow
                flowsdict.setdefault(t,[[],[]]) # first sizes and then times
                # add the values to the lists
                flowsdict[t][0].append(len(packet))
                flowsdict[t][1].append(float(packet.time))
            print('    number of flows is',len(flowsdict))
            
            filename = pcap.split('\\')[-1] # the file name of the pcap
            label = int(iddf[iddf['fname']==filename]['label'].iloc[0])
            for flow in flowsdict:
                t = flowsdict[flow]
                if len(t[0]) >= 20:
                    a = list(chunks(t[0],20))
                    b = list(chunks(t[1],20))
                    t = [[a[i],b[i]] for i in range(len(a)) if len(a[i])>=20]
                else:
                    continue
                DATA.extend(t)
                LABELS.extend([label]*len(t))
    x = np.array(DATA)
    y = np.array(LABELS)
    np.save('data_.npy',x)
    np.save('labels_.npy',y)
    
    
            
            
                
    
    
 
            
    
if __name__ == "__main__":
    main()