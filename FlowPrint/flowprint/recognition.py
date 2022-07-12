import argparse
import os
import numpy as np
import csv
import pandas as pd

from preprocessor import Preprocessor
from flowprint import FlowPrint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def rename_files():
    mypath = "C:/Users/yoel2/Downloads/filtered_raw_dataset_temu2016_first_10_min/Mixed/"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for f in onlyfiles:
        old_name = f
        new_name = f[0:-24]
        try:
            os.rename(mypath + old_name, mypath + new_name)
        except FileExistsError:
            print("File already Exists")
            print("Removing existing file")
            # skip the below code
            # if you don't' want to forcefully rename
            os.remove(new_name)
            # rename it
            os.rename(old_name, new_name)
            print('Done renaming a file')
            print("finished")



if __name__ == "__main__":
    # read_csv()

    ########################################################################
    #                             Handle input                             #
    ########################################################################
    # Parse input
    parser = argparse.ArgumentParser("FlowPrint recognition example")
    parser.add_argument('--files', nargs='+', help='files to use as input')
    parser.add_argument(
        '--dir', help='directory containing files to use as input')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='train ratio of data (default=0.5)')
    parser.add_argument('--random', action='store_true',
                        help='split randomly instead of sequentially')
    args = parser.parse_args()

    # Check if arguments were given
    if (args.files is None and args.dir is None) or\
       (args.files is not None and args.dir is not None):
        raise ValueError(
            "Please specify either --files or --dir but not both.")

    ########################################################################
    #                              Read data                               #
    ########################################################################
    # Create preprocessor
    preprocessor = Preprocessor(verbose=True)
    # Process all files

    sub_directories = [x[0] + "/" for x in os.walk(args.dir)]
    sub_directories = sub_directories[1:len(sub_directories)]
    # Get file names
    files = args.files or [args.dir+x for x in os.listdir(args.dir)]


# do not delete this - for the 400GB
    # files = []
    # labels = []
    # for dir in sub_directories:
    #     for x in os.listdir(dir):
    #         # sessions
    #         if dir[-8:len(dir)] != "sessions":
    #             print(dir[-8:len(dir)])
    #             files.append(dir + x)
    #             labels.append(os.path.basename(os.path.normpath(dir)))

    files = []
    labels = []
    data= pd.read_csv("C:/Users/yoel2/Downloads/filtered_raw_dataset_temu2016_first_10_min/file2labels.csv")
    for i in range(0, len(data.filename)):
        files.append(args.dir + data.filename[i])
        labels.append(data.os[i] + " " + data.browser[i] + " " + data.domain[i])

    # labels = files
    # labels = []
    # labels = list(([x.split("_")[0] for x in files]))
    print(str(labels))
    print()
    X, y = preprocessor.process(files, labels)

    ########################################################################
    #                              Split data                              #
    ########################################################################
    if args.random:
        # Perform random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.ratio, random_state=42)

    # Perform temporal split split
    else:
        # Initialise training and testing data
        X_train = list()
        y_train = list()
        X_test = list()
        y_test = list()

        # Loop over each different app
        for app in np.unique(y):
            # Extract flows relevant for selected app
            X_app = X[y == app]
            y_app = y[y == app]

            # Create train and test instances by split
            X_app_train = X_app[:int(X_app.shape[0]*args.ratio)]
            y_app_train = y_app[:int(X_app.shape[0]*args.ratio)]
            X_app_test = X_app[int(X_app.shape[0]*args.ratio):]
            y_app_test = y_app[int(X_app.shape[0]*args.ratio):]

            # Append to training/testing data
            X_train.append(X_app_train)
            y_train.append(y_app_train)
            X_test.append(X_app_test)
            y_test.append(y_app_test)

            # Print how we split the data
            print("Split {:40} into {} train and {} test flows".format(
                app, X_app_train.shape[0], X_app_test.shape[0]))

        # Concatenate
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

    ########################################################################
    #                              Flowprint                               #
    ########################################################################
    # Create FlowPrint example
    flowprint = FlowPrint(
        batch=300,
        window=30,
        correlation=0.1,
        similarity=0.9
    )

    # Fit FlowPrint with training data
    flowprint.fit(X_train, y_train)
    # Create test fingerprints

    fp_test = flowprint.fingerprinter.fit_predict(X_test)
    # Create prediction
    y_pred = flowprint.recognize(fp_test)

    ########################################################################
    #                           Print evaluation                           #
    ########################################################################

    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    num = len(X_train) + len(X_test)
    print("\n" + str(num))
