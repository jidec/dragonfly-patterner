import pandas as pd

def preprocessRecords(csv_names, id_cols, csv_seps, proj_root="../.."):
    """
        Preprocess records data, adding cols for recordID and dataSource that are important for image filtering later
        :param list<str> csv_names: a list of names of csvs within the data folder to merge into a single all_records.csv
        :param list<str> id_cols: a list of columns specifying which respective columns in the csvs to draw record IDs from
        :param str proj_root: the location of the project folder i.e. dragonfly-patterner containing an /R/src/preprocessiNat.R script to source from
    """

    # for every csv
    for csv_name, id_col, sep in tuple(zip(csv_names,id_cols,csv_seps)):

        # read in csv
        df = pd.read_csv(filepath_or_buffer=proj_root + "/data/" + csv_name + ".csv", sep=sep,index_col=None)
        print("Read " + csv_name + "...")

        # create new col for recordID
        df['recordID'] = df[id_col]
        print("Added recordID column...")

        # create new col for the dataSource i.e. 'antweb', 'inat', 'odonatacentral'
        df['dataSource'] = [csv_name.split("_")[0]] * df.shape[0]
        print("Added dataSource column...")

        # write csv
        df.to_csv(proj_root + "/data/" + csv_name + ".csv")
        print("Wrote " + csv_name + " - finished!")

