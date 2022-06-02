import pandas as pd

# note - doesn't work yet, just sketch
# should also merge different record sets
def preprocessMergeRecords(csv_names,id_cols,proj_root="../.."):
    """
        Preprocess and merge records data,
        :param list<str> csv_names: a list of names of csvs within the data folder to merge into a single all_records.csv
        :param list<str> id_cols: a list of columns specifying which respective columns in the csvs to draw record IDs from
        :param str proj_root: the location of the project folder i.e. dragonfly-patterner containing an /R/src/preprocessiNat.R script to source from
    """

    # merge record dataframes
    # create empty dataframe
    df = pd.DataFrame()
    # for every csv
    for name,id_col in tuple(csv_names,id_cols):
        # read in csv
        next_df = pd.read_csv(filepath_or_buffer=proj_root + "/" + name + ".csv") #, header=True, sep="\t",quotechar="")
        df['recordID'] = df[id_col]
        df = pd.concat(df, next_df)

    #df['numImages'] = df['mediaType'].count("StillImage")

    #keeps = ["occurrenceID","family","genus","species","infraspecificEpithet",
    #         "decimalLatitude","decimalLongitude","coordinateUncertaintyInMeters",
    #         "eventDate","day","month","year","taxonKey",
    #         "catalogNumber","identifiedBy","recordedBy","issue","numImages"]
    #df = df[keeps]

    #df = df[df['numImages'] != 0,:]

    df.to_csv(proj_root + "/data/all_records.csv")

