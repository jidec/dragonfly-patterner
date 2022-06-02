import pandas as pd

def writeiNatGenusList(inat_csv_name,proj_root="../.."):
    """
        Write a list of genera to data folder for use by downloader scripts
    """

    df = pd.read_csv(filepath_or_buffer=proj_root + "/data/" + inat_csv_name + ".csv")
    genera = pd.Series(df['genus'].unique())
    # make sure this works with downloader
    genera.to_csv(proj_root + "/data/other/genus_list.csv")