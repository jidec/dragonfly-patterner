import requests
import json
import pandas as pd
from sourceRdefs import getFilterImageIDs
from os import listdir
import time
import datetime

def downloadiNatRandomSingleImages(n,proj_root="../.."):
    """
        Download random single images - typically used for creating generalized training sets

        :param str genus: the genus name to fix image names for
        :param str proj_root: the path to the project folder
    """
    print("Loaded all observation info")
    data = pd.read_csv(proj_root + '/data/inat_data.csv',sep=',')

    # WIP implementation to download only non-downloaded images
    #existing_image_ids = pd.DataFrame(listdir(pylib_root + '/../../../data/all_images'))
    #existing_image_ids.columns = ['catalogNumber']
    #existing_image_ids.str.split(',')

    print("Picked " + str(n) + " random observations")
    ids = data.sample(n)['catalogNumber']
    print("Downloading observations...")
    for index, i in enumerate(ids):

        # every 300 observations, sleep for an hour to avoid throttling by iNat server
        if index % 300 == 0 and index != 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Downloaded 300 images, waiting one hour until next 300 starting at " + current_time)
            time.sleep(3600)
        try:
            link = 'https://api.inaturalist.org/v1/observations/' + str(i)
            x = requests.get(link)
            obs = json.loads(x.text)

            # parse down to image url
            result = obs.get("results")
            result = result[0]
            result.keys()
            result = result.get('observation_photos')
            result = result[0]
            result = result.get('photo')
            result = result.get('url')

            # replace square with original to get full size
            result = result.replace("square", "original")
            img = requests.get(result).content
            file = open(proj_root + "/data/all_images/INATRANDOM-" + str(i) + ".jpg", "wb")

            # write file
            if index%10 == 0: print(str(index))

            file.write(img)
            file.close()
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed at ' + link + ' probably due to throttling by the iNat server')

    print("Finished downloading observations")