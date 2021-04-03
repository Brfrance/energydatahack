import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from pathlib import Path
from copy import deepcopy

plt.close("all")

# convert all data files to csv format
def convertData():
    p = 'data_set/'
    files = sorted(Path(p).iterdir(), key=path.getmtime, reverse=False)
    for file in files:
        if 'MIN' in file.stem:
            tocsv(file)

# convert a data file to csv format and save in data_set/csv/
def tocsv(path):
    file = open(path, 'r')
    print('data_set/csv/' + path.stem + '.csv')
    csv = open("data_set/csv/" + path.stem + '.csv', "w")
    csv.write('date,t1,t2,t3,t4,t5,t6,avg\n')
    for line in file:
        line = line.split()
        avg = (sum([int(x) for x in line[2:]]) / 6)

        line = "{0} {1},{2},{3},{4},{5},{6},{7},{8}"\
            .format(line[0], line[1], line[2], line[3], line[4], line[5],\
            line[6], line[7], avg)
        csv.write(line + '\n')

    file.close()
    csv.close()

# create a dictionary of label: [files]
# label = type;function
def labelise():
    dic = {}
    p = 'data_set/'
    files = sorted(Path(p).iterdir(), key=path.getmtime, reverse=False)
    for file in files:
        if '_label' in file.stem:
            f = open(file, 'r')
            label = f.read()
            if label not in dic:
                dic[label] = []

            dic[label].append(file.stem[:14])

    return dic

# plot csv files, can be filtered by type
def parseDf(labelType=None):
    fig, ax = plt.subplots()
    dataframes = {}
    labels = labelise()
    for k in labels.keys():
        for dataId in labels[k]:
            if labelType and k.split(';')[0] != labelType:
                continue

            df = pd.read_csv('data_set/csv/MIN ' +  dataId  + '.csv')

            if k not in dataframes:
                dataframes[k] = df
            else:
                dataframes[k] = dataframes[k].append(df, ignore_index=True)

    for k in labels.keys():
        if labelType and k.split(';')[0] != labelType:
                continue

        df = dataframes[k]
        df['date']= pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
        df = df.groupby((df['date'].dt.dayofweek) * 24 + (df['date'].dt.hour))\
            .mean()
        df = df.rename(columns={"avg": k})
        pl = df.plot(ax=ax, y=k)
        pl.set_xlabel('Hour of week')
        pl.set_ylabel('kWh')
    plt.xticks(range(0, 170, 4))
    plt.show()


parseDf()
