import pandas as pd

def writeiNatGenusList(inat_csv_name,proj_root="../.."):
    """
        Write a list of genera to data folder for use by downloader scripts
        The used .csv must have a column called "genus"

        :param str inat_csv_name: the name of the .csv to write a genus list from, NOT containing the .csv suffix
    """

    df = pd.read_csv(filepath_or_buffer=proj_root + "/data/other/raw_records" + inat_csv_name + ".csv")
    genera = pd.Series(df['genus'].unique())
    # make sure this works with downloader
    genera.to_csv(proj_root + "/data/other/genus_list.csv")