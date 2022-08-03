import os
import pandas as pd
from os.path import exists
from imageIDToRecordID import imageIDToRecordID

def getIDsFromDir(direct_dir, contains):
    image_ids = [s.replace('.jpg', '') for s in os.listdir(direct_dir)]
    image_ids = [i for i in image_ids if contains in i]
    return(image_ids)