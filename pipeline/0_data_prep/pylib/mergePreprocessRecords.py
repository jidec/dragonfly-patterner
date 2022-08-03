import pandas as pd

def mergePreprocessRecords(raw_records_csv_names, id_cols, id_prefixes, csv_seps, proj_root="../.."):
    """
        Concatenate records .csvs placed in data/other/raw_records into a single data/records.csv file
        Delimits by commas and adds new cols for recordID
        The name of the .csv is added to the dataSource field

        :param list<str> csv_names: a list of names of .csvs within the data folder, NOT including the .csv suffix
        :param list<str> id_cols: a list of columns specifying which respective columns in the csvs to draw record IDs from
        :param str proj_root: the location of the project folder
    """
    # start with an empty
    merged = None

    # for every csv
    for csv_name, id_col, sep, id_pre in tuple(zip(raw_records_csv_names,id_cols,csv_seps,id_prefixes)):

        # read in csv
        df = pd.read_csv(filepath_or_buffer=proj_root + "/data/other/raw_records/" + csv_name + ".csv", sep=sep, index_col=None)
        print("Read " + csv_name + "...")

        # create new col for recordID
        ids = list(df[id_col])
        df['recordID'] = [id_pre + '-' + str(i) for i in ids]
        print("Added recordID column...")

        # create new col for the dataSource i.e. 'antweb', 'inat', 'odonatacentral'
        df['dataSource'] = [csv_name] * df.shape[0]
        print("Added dataSource column...")

        # write csv
        #df.to_csv(proj_root + "/data/" + csv_name + ".csv")
        #print("Wrote " + csv_name + " - finished!")

        if merged is None:
            merged = df
        else:
            #merged = merged.merge(df)
            merged = pd.concat([merged,df])
        print("Merged into new records")
        print(len(merged))
    merged.to_csv(proj_root + "/data/records.csv")

