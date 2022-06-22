#imports
from glob import glob
from os.path import isdir
from pandas import read_csv,read_excel
import numpy as np
import pandas
from requests import head
from torch import classes
from json import loads



applabels = {1:'dropbox',2:'facebook',3:'google',4:'microsoft',5:'teamviewer',6:'twitter',7:'youtube',8:'other'}
browserslabels = {1:'chrome',2:'firefox',3:'IExplorer',4:'Safari'}
osesslabels = {1:'Linux',2:'Windows',3:'OSX'}
def classify(subclass,id):
    if subclass == "apps":
        return applabels.get(id)
    elif subclass == 'browsers':
        return browserslabels.get(id)
    elif subclass == 'osess':
        return osesslabels.get(id)
    elif subclass == 'boa':
        t = str(id)
        return f"{classify('apps',int(t[1]))}_{classify('browsers',int(t[2]))}_{classify('osess',int(t[4]))}"
    else:
        raise ValueError("non existent subclass gievn")

def precission(m,classes,subclass):
    s = []
    for i in range(len(classes)):
        index = m[i][i]
        support = np.sum(m[i,:])
        t = np.sum(m[:,i])
        if t > 0:
            s.append((classify(subclass,int(classes[i])),index/t,support))
            print('precission',classify(subclass,int(classes[i])),index/t,support)
        else:
            s.append((classify(subclass,int(classes[i])),0,support))
            print('precission',classify(subclass,int(classes[i])),0,support)
    return s

def recall(m,classes,subclass):
    s = []
    for i in range(len(classes)):
        support = np.sum(m[i,:])
        index = m[i][i]
        if support > 0:
            s.append((classify(subclass,int(classes[i])),index/support, support))
            print('recall',classify(subclass,int(classes[i])),index/support,support)
        else:
            s.append((classify(subclass,int(classes[i])),0, support))
            print('recall',classify(subclass,int(classes[i])),0,support)
    return s

def f1(m,classes,subclass):
    recalls = recall(m,classes,subclass)
    precissions = precission(m,classes,subclass)
    s = []
    for i in range(len(classes)):
        r,p = recalls[i][1],precissions[i][1]
        if r+p > 0:
            s.append((classify(subclass,int(classes[i])), (r*p*2) / (r+p), recalls[i][2]))
            print('F1',classify(subclass,int(classes[i])), (r*p*2) / (r+p), recalls[i][2])
        else:
            s.append((classify(subclass,int(classes[i])), 0, recalls[i][2]))
            print('F1',classify(subclass,int(classes[i])), 0, recalls[i][2])
    return s,recalls,precissions




def main():
    # print(glob("./*"))
    INDEX = 2
    SUBCLASS = ["OS","BROWSERS","APPS","BOA"]
    MODEL = 'ODE-FLOW'
    
    # df = read_csv("./results/cgnn/results_boa.csv")
    resultsdf = read_csv("./results/results.csv")
    # classes = [int(i) for i in list(df.columns)[1:]]
    # classes.sort()
    # # print(classes)
    # m = df.to_numpy()
    # m = np.delete(m,0,1)
    # lines = []
    # data = f1(m,classes,'boa')
    # for i in range(len(classes)):
    #     lines.append([MODEL,SUBCLASS[INDEX],data[0][i][0],data[2][i][1],data[1][i][1],data[0][i][1],data[0][i][2]])
    # tdf = pandas.DataFrame(lines,columns=resultsdf.columns[1:])
    # resultsdf = pandas.concat([resultsdf,tdf],ignore_index=True)
    # resultsdf = resultsdf.drop([resultsdf.columns[0]],axis=1)
    # resultsdf.to_csv("./results/results.csv")
    # classes = loads('''[11102, 12101, 12201, 13101, 13102, 13103, 13201, 13202, 13302,
    #    13403, 14102, 14202, 14302, 16101, 16102, 16103, 16201, 16202,
    #    16302, 16403, 17101, 17201, 17403, 18101, 18102, 18201, 18202,
    #    18302, 18403]''')
    
    s = '''1       0.00      0.00      0.00         0
           2       0.00      0.00      0.00         0
           3       0.89      0.64      0.74      1580
           4       0.13      0.70      0.22        10
           6       0.85      0.94      0.89      2819
           7       0.86      0.77      0.81       370
           8       0.35      0.48      0.40       300'''
    data = [[float(j) for j in i.split(' ') if j ] for i in s.split('\n')]
    lines = []
    for i in data:
        i[0] = classify('apps',int(i[0]))#classes[int(i[0])])
        lines.append( [MODEL,SUBCLASS[INDEX]]+i)
    tdf = pandas.DataFrame(lines,columns=resultsdf.columns[1:])  
    resultsdf = pandas.concat([resultsdf,tdf],ignore_index=True)
    resultsdf = resultsdf.drop([resultsdf.columns[0]],axis=1)
    resultsdf.to_csv("./results/results.csv")

if __name__ == "__main__":
    main()
    




