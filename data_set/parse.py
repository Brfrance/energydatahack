import os
import fnmatch
import pandas as pd


def get_chorus(RAE, data_set_link_chorus_rae):

    chorus = data_set_link_chorus_rae[(data_set_link_chorus_rae.RAE == RAE)].chorus

    return chorus.values[0]

def get_type_fct_building(chorus, data_set_inventory):


    print(type(chorus))
    frame = data_set_inventory[(data_set_inventory["Code Chorus"].str.contains(chorus))]

    if frame.empty:
        return None

    fct_building = frame.iloc[0]["Fonction"]
    type_building = frame.iloc[0]["Type"]

    return (type_building, fct_building)

def write_label_file(RAE, labels):

    f = open(RAE + "_label.txt", "w")

    f.write(labels[0] + ";" + labels[1] + "\n")

    f.close()

def create_label_files(data_set_inventory, data_set_link_chorus_rae):
    
    files = [f for f in os.listdir('.') if (os.path.isfile(f) and fnmatch.fnmatch(f, "MIN *"))]

    print(files)

    for filename in files:

        s = filename.split(" ")
        s = s[1].split('.')
        RAE = s[0]

        chorus = get_chorus(int(RAE), data_set_link_chorus_rae)
        print(chorus)

        if not chorus or chorus == "NC":
            continue

        labels = get_type_fct_building(str(chorus), data_set_inventory)
        print(labels)

        if labels == None:
            continue

        write_label_file(RAE, labels)

        




if __name__ == "__main__":

    #recup data set
    data_set_link_chorus_rae = pd.read_excel("data_set_link_chorus_rae.xls")
    data_set_inventory = pd.read_csv("inv-imm-20181231.csv", delimiter=";")

    create_label_files(data_set_inventory, data_set_link_chorus_rae)

