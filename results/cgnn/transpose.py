from hashlib import new
from nbformat import read
from pandas import read_csv,DataFrame
import numpy as np



def main():
    boa_labels = [11102, 11202, 12101, 12201, 13101, 13102, 13103, 13201, 13202, 13302, 13403, 14101, 14102, 14202, 14302, 15102,
                  15202, 15302, 16101, 16102, 16103, 16201, 16202, 16302, 16403, 17101, 17201, 17403, 18101, 18102, 18103, 18201, 18202, 18302, 18403]
    apps_labels  = list(range(8))
    browsers_labels = list(range(4))
    os_labels = list(range(3))
    df = read_csv('results\cgnn\\CGNN-OS.csv').to_numpy()

    newdf = np.zeros(df.shape)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            newdf[i][df.shape[1]-1-j] = df[i][j]
    newdf = DataFrame(newdf,columns=os_labels)
    newdf.to_csv('results\cgnn\\results_osses.csv')
    
if __name__ == "__main__":
    main()
    


