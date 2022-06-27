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
from multiprocessing import Pool
import tqdm
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process
from concurrent.futures import ThreadPoolExecutor

def downloadAntWebImages(start_index,end_index,img_size="med",shot_types=['h','d'],proj_root='../..'):
    """
       Download all AntWeb images using GBIF catalogNumbers

       :param int start_index: the row index from the gbif dataframe to start at
       :param int end_index: the row index from the gbif dataframe to end at
       :param str proj_root: the path to the project folder
    """

    # load AntWeb data downloaded from GBIF
    df = pd.read_csv(proj_root + "/data/antweb_records.csv")
    urls_ids = []

    # setup queries
    session = requests.Session()
    retry = requests.adapters.Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    for index, row in df.iterrows():
        if index >= start_index and index <= end_index:
            antweb_id = df.loc[:, "catalogNumber"][index]

            # query to verify that specimen isn't fossil (can't do this from raw gbif data unfortunately)
            url = 'https://www.antweb.org/v3.1/specimens?specimenCode=' + antweb_id + '&up=1' + '&fossil=false'
            urls_ids.append((url,antweb_id))


    def downloadAntImage(urls_ids):
        r = requests.get(urls_ids[0])
        print("Request 1 url " + urls_ids[0])
        try:
            j = r.json()

            # if the non-fossil query is greater than 0, specimen is not a fossil
            if j['metaData']['count'] > 0:

                # query info for all AntWeb images for that specimen
                url = 'https://www.antweb.org/v3.1/images?specimenCode=' + urls_ids[1] + '&up=1'  # shotType=H
                print("Request 2 url " + url)
                r = requests.get(url)

                # convert image query to json
                try:
                    j2 = r.json()
                    j2 = j2['images'][0]['images']

                    # for every imaging session, each typically imaging a different angle
                    for imaging_session in j2:
                        # continue if we are downloading this shot type
                        type = imaging_session['shotType']
                        if type in shot_types:
                            for url in imaging_session['urls']:
                                # continue if we are downlading this image size
                                if img_size in url:
                                    print(url)
                                    # query for the image
                                    r = requests.get(url)

                                    # open image
                                    img = Image.open(BytesIO(r.content))

                                    # replace dash in AntWeb id (REMEMBER THIS)
                                    antweb_id = antweb_id.replace("-", "")

                                    # filter against greyscale (SEM images)
                                    imgarr = np.array(img)
                                    if len(imgarr.shape) == 3:
                                        # save image to data
                                        img.save(proj_root + "/data/all_images/AW-" + antweb_id + "-" + type + ".jpg")
                except ValueError:
                    print('Decoding JSON has failed at 2nd step, skipping image')
        except json.decoder.JSONDecodeError:
            print('Decoding JSON has failed at first step, skipping image')

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(downloadAntImage, urls_ids)  # urls=[list of url]
    #pool = Pool()
    #pool.map(downloadAntImage, urls_ids)
    #for _ in tqdm.tqdm(pool.map(downloadAntImage, urls_ids), total=len(urls_ids)):
    #    pass
