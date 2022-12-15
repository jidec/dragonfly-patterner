import os
import pandas as pd
from os.path import exists
from imageIDToRecordID import imageIDToRecordID
from os import listdir
from os.path import isfile, join

def getIDsFromFolder(dir):
    """
        Get image ids from a folder
    """
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    ids = [f.split("_")[0] for f in files]
    ids.pop()

    return(ids)