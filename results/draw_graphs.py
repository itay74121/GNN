import matplotlib.pyplot as plt
from pandas import read_csv







def main():
    df = read_csv('./results/results.csv')
    subclasses = df.subclass.unique()    
    category = subclasses[0]
    mid = df[df.subclass == category].copy()
    res = mid.groupby(['class'])
    for i in res:
        d = list(i[1].f1)
        plt.hist([d,[1,2,3]],label=['mapgraph','cgnn'])
        plt.show()
    # plt.show()
        
        
if __name__ == "__main__":
    main()