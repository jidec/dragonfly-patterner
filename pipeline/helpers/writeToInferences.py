import pandas as pd
from os.path import exists

def writeToInferences(inferences,proj_dir):
    """
        Add a new set of infer data to data/inferences.csv

        :param pd.DataFrame inferences
        :param str proj_dir
    """

    # write to inferences.csv if it exists
    if exists(proj_dir + "/data/inferences.csv"):
        current_infers = pd.read_csv(proj_dir + "/data/inferences.csv")
        inferences = pd.concat([current_infers, inferences])

    inferences.to_csv(proj_dir + "/data/inferences.csv",index=False)
