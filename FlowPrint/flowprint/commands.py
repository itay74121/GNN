import os
from os import listdir
from os.path import isfile, join

mypath = "C:\\Users\\yoel2\\Downloads\\flowpic_extracted_flows\\iscx_chat.raw\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

for f in onlyfiles:
    output_fingerprint_file = "./fingerperints/" +  f + ".json"
    os.system("python ./__main__.py --fingerprint " + output_fingerprint_file + "-p " + "\"" +  mypath + f + "\"")

# os.system('cls')
# os.system('python .\__main__.py -h')