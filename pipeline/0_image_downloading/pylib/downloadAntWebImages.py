import pandas as pd
import os
import subprocess
import time
from datetime import datetime
import shutil
import requests
from requests.adapters import HTTPAdapter
from requests.adapters import Retry
import json
from io import BytesIO
import colorsys
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import imghdr
import math
import random

def downloadAntWebImages(start_index,end_index,full_size=False,proj_root='../..'):
    """
       Download all AntWeb images using GBIF catalogNumbers

       :param int start_index: the row index from the gbif dataframe to start at
       :param int end_index: the row index from the gbif dataframe to end at
       :param str proj_root: the path to the project folder
    """

    # load AntWeb data downloaded from GBIF
    df = pd.read_csv(proj_root + "/data/antweb_records.csv")
    for index, row in df.iterrows():
        if index >= start_index and index <= end_index:
            antweb_id = df.loc[:, "catalogNumber"][index]

            # query info for all AntWeb images for that specimen
            session = requests.Session()
            retry = requests.adapters.Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            url = 'https://antweb.org/v3.1/images?specimenCode=' + antweb_id + '&up=1' #shotType=H
            r = requests.get(url)

            # convert image query to json
            d = r.json()
            p = d['images']
            j = json.dumps(p)
            print(j)

            # get a list of links to every medium non sem images
            # for image in list
            #   download image to data as ANTWEB-<antweb_id>-<DORSAL,VENTRAL,ETC...>

            # later, train seg model
            # simple script to get avg color from segment
            # script to normalize images to grey standard - just get background




