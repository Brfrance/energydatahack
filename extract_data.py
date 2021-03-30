import csv
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd


def extract_data_from_txt(file, verbose=False):
    """
    Extract data from file
    """

    indexes = []
    values = []

    with open(file, 'r') as in_file:
        data_iter = csv.reader(
            in_file,
            delimiter='\t'
        )

        for row in data_iter:

            current_date = datetime.strptime(f"{row[0]} {row[1]}", '%d/%m/%Y %H:%M')        
            
            for i in range(2, 8):
                if i > 2:
                    current_date += timedelta(minutes=10)

                indexes.append(current_date)
                values.append(row[i])

    df = pd.DataFrame({
        'Date': indexes,
        'Value': values
    })

    if verbose:
        df_subset = df[df['Date'] >= pd.Timestamp(2014,12,31,0,0,0)]
        plt.plot(df_subset['Date'], df_subset['Value'])
        plt.show()

    return df


def extract_label_from_txt(filename):
    """Get building type and building function."""
    with open(filename, 'r') as in_file:
        txt = in_file.readline()
        split = txt.split(";")
    return split[0], split[1][:-1]


def get_labels(filename):
    """Get all building types and building functions."""
    with open(filename, 'r') as in_file:
        read = in_file.readline()
    return read.split(";")
