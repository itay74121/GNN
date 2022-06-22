from glob import glob
from os.path import isdir
from PcapTocsv import pcaps2csvs
import pandas as pd
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def main():
    dirs = [i for i in glob('./mapgraph/preprocess/samsung*') if isdir(i)]
    dirs.sort()
    print(dirs)
    for d in dirs:
        edges = []
        features = [] 
        labels = []
        files = glob(d+'\\*.csv')
        files.sort()
        # csvs = pcaps2csvs(files,20,[0]*len(files))
        pairs = list(chunks(files,2))
        e_col = None
        f_col = None
        for i,p in enumerate(pairs):
            e = pd.read_csv(p[0])
            e_col = e.columns
            e['id'] = i
            e = e.to_numpy()[:,1:]
            f = pd.read_csv(p[1])
            f_col = f.columns
            f['graph_id'] = i
            f = f.to_numpy()[:,1:]
            edges.extend(e)
            features.extend(f)
            labels.append([i,dirs.index(d)])
        edges = pd.DataFrame(edges,columns = e_col[1:])
        features = pd.DataFrame(features,columns = f_col[1:])
        labels = pd.DataFrame(labels,columns=['id','label'])
        edges.to_csv(d+"\\edges.csv")
        features.to_csv(d+"\\features.csv")
        labels.to_csv(d+"\\labels.csv")
        
if __name__ == "__main__":
    main()