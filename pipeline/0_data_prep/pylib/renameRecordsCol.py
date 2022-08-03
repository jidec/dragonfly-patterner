import pandas as pd

def renameRecordsCol(records_name,id_col="collectioncode",proj_dir="../.."):
    df = pd.read_csv(proj_dir + "/data/" + records_name + ".csv", delim_whitespace=True)
    df = df.rename(columns={id_col: 'recordID'}, inplace=True)
    df.to_csv(proj_dir + "/data/" + records_name + ".csv")